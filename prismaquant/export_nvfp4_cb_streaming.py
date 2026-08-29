"""Streaming NVFP4-CB / FP8-CB exporter for 200-300B-class models.

Sibling of :mod:`prismaquant.export_nvfp4_cb` (the in-memory exporter) built
for models whose full weights do not fit resident: Hy3 (~557 GB, bf16) and
DSv4-Flash (~295 GB, fp8-native). The in-memory exporter's ``_load_skeleton``
materialises EVERY shard into one dict and accumulates EVERY output tensor
before writing (dsv4_readiness.md gap 2) — a full-model materialisation twice
over. This exporter streams instead:

  * lazy shard index (``_LazySkeleton``): one source tensor resident at a time
    (the ``export_gguf_direct._ShardIndex`` pattern), with fp8-block
    dequant-on-read for native-fp8 sources (``layer_streaming``);
  * per-expert -> stacked bridging: MoE experts stored per-expert on disk
    (Hy3: ``…experts.{i}.{gate,up,down}_proj``) are packed one expert at a
    time and the SMALL packed byte-rows stacked — never all experts resident;
  * two-pass streaming safetensors write (``_StreamWriter``): sizes are
    computed analytically in pass 1 (CB type_size / source metadata, no data
    load), the header is written, then each tensor is produced+written+freed
    in pass 2. Peak residency ~= one source tensor + the codebooks.

The PACKED BYTES are identical to the in-memory exporter (both call
``cb.nvfp4_cb_pack``; CB scales are per-expert/per-row/per-group, so packing
one expert alone equals packing it inside the stack) — pinned byte-for-byte
in tests/test_nvfp4_cb_streaming.py. Both exporters call the single
``cb_export_config`` builder for container/config/sidecar metadata.

Scope: bf16 source + fp8-source READ + CB families + BF16 passthrough + the
SOURCE-PASSTHROUGH family (``allocator_candidates.SOURCE_PASSTHROUGH_CONTRACTS``
— FP8_SOURCE normalized into the compressed-tensors namespace, MXFP4_SOURCE and
FP8_BLOCK_UE8M0_SOURCE copied byte-verbatim under the checkpoint's own names)
+ stock-CT **DENSE** rungs (vanilla NVFP4 / FP8_DYNAMIC
quantised in-container). Stock rungs are packed RTN via the authoritative
``export_native_compressed`` codecs (byte-identical to the in-memory
export_nvfp4_cb and to those packers called directly; no GPTQ/act-order in this
lane — the CB cost stage measures stock rungs RTN-grade). Their on-disk sizes
are ANALYTIC so the streaming header needs no pack:

  * NVFP4  -> ``weight_packed`` uint8 [N, K/2] + ``weight_scale`` fp8_e4m3
    [N, K/16] + ``weight_global_scale`` fp32 [1] + ``input_global_scale`` fp32 [1]
  * FP8_DYNAMIC -> ``weight`` fp8_e4m3 [N, K] + ``weight_scale`` fp32 [N, 1]

Stock rungs on MoE **expert stacks** are NOT streamed: the CB container's stock
config emits a packed-name regex that vLLM's MoE dispatch cannot match to its
per-expert probes, and the CT codec is 2-D — an expert stack assigned a stock
format hard-fails with a pointer to constrain the allocator (put experts on a
CB rung / FP8_SOURCE / BF16; the dense tier is where vanilla formats win). The
config_groups for stock rungs use the EXACT compressed-tensors vocabulary (no
``"scheme"`` key) under the vLLM-internal target name so the plugin delegates
them to CompressedTensorsConfig.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import inspect
import json
import os
import pickle
import queue
import re
import struct
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from collections import Counter
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import torch
from safetensors import safe_open

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.shard_layout import (
    DEFAULT_SHARD_BYTES,
    SHARD_INDEX_NAME,
    SHARD_NAME_RE,
    SINGLE_CONTAINER_NAME,
    container_names,
    plan_shards,
    tensor_payload_identity,
    write_shard_index,
)
from prismaquant.allocator_candidates import (
    ROUTE_PENDING_PASSTHROUGH_FORMATS,
    SOURCE_PASSTHROUGH_CONTRACTS,
)
from prismaquant.cb_export_config import (
    PER_EXPERT_FORMAT_GROUPS_KEY,
    PER_EXPERT_FORMAT_GROUPS_VERSION,
    SOURCE_PASSTHROUGH_EXPORT_FORMATS,
    STREAMING_REQUANT_EXPORT_FORMATS,
    parse_source_passthrough_declaration,
    build_cb_scheme,
    build_quant_config,
    build_quantized_embedding_declaration,
    cb_scheme_reuse_signature,
    codebook_tensor_names as _codebook_tensor_names,
    codebook_tensors as _codebook_tensors,
    source_passthrough_wire,
    source_passthrough_wire_id,
)
from prismaquant.format_registry import (
    canonical_format_name,
    get_format as _fr_get_format,
)
from prismaquant.cb_route_status_gate import (
    NON_NATIVE_TARGET_ENV,
    ROUTE_OVERRIDE_ENV,
)
from prismaquant.export_nvfp4_cb import (
    _canonical_qname,
    _export_base_name,
    _git_commit,
    _parse_cb_format,
    _role_of,
    _to_device,
    _try_resolve_direct_packed_expert,
    _try_resolve_skeleton,
)
from prismaquant.layer_config import canonicalize_assignment, load_assignment
from prismaquant.model_profiles import detect_profile
from prismaquant.routed_moe_codebooks import (
    ROUTED_BOOK_KEYING_ROLE,
    ROUTED_BOOK_KEYING_STACK,
    ROUTED_MOE_CBL_BANK_RUNGS,
    RoutedMoECodebookRole,
    bundle_role_qname,
    bundle_stack_qname,
    describe_split_book_refusal,
    fused_targets_with_split_books,
    logical_role_qname,
    split_role_rows,
    stacked_role_col_weights,
)
from prismaquant.export_output_safety import (
    prepare_fresh_export_directory,
    transactional_directory_output,
)
from prismaquant.dspark_source_metadata import (
    apply_dspark_overlay_to_model_config,
    apply_dspark_overlay_to_quant_config,
    build_dspark_target_bridge,
    discover_dspark_source_overlay,
    dspark_cb_construction_target_for_physical_output,
    dspark_cb_expected_physical_targets,
    dspark_cb_physical_output_for_recipe_target,
    dspark_cb_physical_output_for_construction_target,
    dspark_cb_physical_source_for_recipe_target,
    dspark_cb_source_passthrough_mapping,
    dspark_construction_unit_for_physical_target,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    cb_payload_summary,
    cb_serialization_metadata_from_assignment_payload,
    cb_serialization_context_from_env,
    codebook_source_for_format,
    effective_codebook_source_scope,
    scale_sweep_for_format,
    cb_tensor_payload_breakdown,
    finalize_cb_export_artifact_inventory,
    resolve_cb_encode_tier,
    assignment_serialization_sha256,
    whole_artifact_budget_from_assignment_payload,
    assert_exclusions_match_budget_stamp,
    validate_cb_sidecar_tensors,
    validate_cb_assignment_serialization_stamps,
    validate_cb_serialization_context_stamp,
)
from prismaquant.nvfp4_activation_contract import (
    NVFP4_ACTIVATION_CONTRACT_KEY,
    routed_moe_attested_module_names,
    NVFP4_ACTIVATION_CONTRACT_SCHEMA,
    NVFP4_ACTIVATION_EXECUTION,
    build_execution_contract,
    calibrated_input_global_scales_with_sources,
    input_global_scale_tensor,
    resolve_input_global_scale_policy,
)

# safetensors dtype string codes for the tensors we emit.
#
# F8_E8M0 is here because the byte-verbatim passthrough lane ships the
# checkpoint's E8M0 scale plane UNCHANGED: DSv4-Flash carries it for both the
# routed MXFP4 experts (`layers.N.ffn.experts.E.w{1,2,3}.scale`) and the
# block-FP8 body (`layers.N.attn.*.scale`). Re-encoding it to F32, as the
# FP8_SOURCE branch does with its fp32 block scales, would quadruple its size
# and stop being a byte copy. A missing entry is not a soft failure —
# `_StreamWriter.write` indexes this map to build the header and would die with
# a bare KeyError.
_ST_DTYPE = {
    torch.uint8: "U8", torch.float32: "F32", torch.float16: "F16",
    torch.bfloat16: "BF16", torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e8m0fnu: "F8_E8M0",
    torch.int64: "I64", torch.int32: "I32", torch.int8: "I8",
    torch.bool: "BOOL",
}
# Inverse (safetensors dtype string -> torch dtype), used by the DELTA-EXPORT
# reuse path to read a prior artifact's per-tensor dtype from its header.
_ST_DTYPE_INV = {v: k for k, v in _ST_DTYPE.items()}
_LEGACY_EXPERT_RE = re.compile(
    r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

_LEGACY_PACKED_PROJECTIONS = {
    "gate_up_proj": ("gate_proj", "up_proj"),
    "down_proj": ("down_proj",),
    "gate_proj": ("gate_proj",),
    "up_proj": ("up_proj",),
}

_PER_EXPERT_GROUP_DISCRIMINATOR = "format_group_"
_PER_EXPERT_LAYER_RE = re.compile(r"(?:^|[.])layers[.](\d+)(?:[.]|$)")


def _format_group_slug(format_wire_id: str) -> str:
    """Filesystem/tensor-name-safe discriminator for one wire format id."""

    slug = re.sub(r"[^a-z0-9]+", "_", str(format_wire_id).lower()).strip("_")
    if not slug:
        raise ValueError(f"empty format wire id {format_wire_id!r}")
    return _PER_EXPERT_GROUP_DISCRIMINATOR + slug


def _per_expert_format_wire_id(format_name: str) -> str:
    canonical = canonical_format_name(format_name)
    if _parse_cb_format(canonical) is not None:
        return canonical
    if canonical == "MXFP4_SOURCE":
        return source_passthrough_wire_id(canonical)
    raise ValueError(
        "per-expert stack groups support NVFP4_CB/FP8_CB rungs and "
        f"MXFP4_SOURCE, got {format_name!r}"
    )


def _per_expert_layer_id(prefix: str) -> str:
    match = _PER_EXPERT_LAYER_RE.search(prefix)
    if match is None:
        raise ValueError(
            f"{prefix}: cannot derive the numeric layer id required by "
            f"{PER_EXPERT_FORMAT_GROUPS_KEY}"
        )
    return match.group(1)


def _per_expert_family(profile, packed_proj: str) -> str:
    projections = _packed_expert_projection_names(profile, packed_proj)
    if len(projections) == 2:
        return "w13"
    if len(projections) == 1:
        return "w2"
    raise ValueError(
        f"{packed_proj}: proposed per-expert wire contract has only w13/w2 "
        f"families, got projections={projections}"
    )


# ---------------------------------------------------------------------------
# Lazy weight source (one tensor resident at a time; fp8-block dequant-on-read)
# ---------------------------------------------------------------------------

class _LazySkeleton:
    """Shard-indexed lazy safetensors reader. ``__contains__`` matches the
    dict-skeleton contract the export_nvfp4_cb name resolvers expect, but
    tensor data is only touched on ``load``/``dequant_weight`` and shape/dtype
    come from the safetensors metadata (no data load)."""

    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            single = self.dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(
                    f"no model.safetensors[.index.json] under {self.dir}")
            with safe_open(single, framework="pt", device="cpu") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self._open: dict[str, object] = {}
        self._shard_hdr: dict[str, tuple[dict, int]] = {}
        try:
            self._profile = detect_profile(str(self.dir))
        except Exception:
            self._profile = None
        from prismaquant.cb_source_decode import build_cb_source_fp8_scale_map

        # Profile-aware map: legacy `.weight_scale_inv`, DSv4 `.scale`, nested
        # text/multimodal namespaces, and explicitly declared MXFP4 experts all
        # share the exact loader-side decode contract.
        self._fp8_scale_inv_map = build_cb_source_fp8_scale_map(self.dir)

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def keys(self):
        return self.weight_map.keys()

    # Bound concurrently-open shard mmaps. Unbounded handles grew total_vm
    # to ~1TB on the 233-shard Hy3 source and the box global-OOMed with the
    # exporter's CPU RSS ~0 and torch CUDA alloc 0 — the consumer was
    # driver-side pinning tied to live source mappings (2026-07-19).
    _MAX_OPEN_SHARDS = 4

    def _handle(self, name: str):
        shard = self.weight_map[name]
        if shard not in self._open:
            while len(self._open) >= self._MAX_OPEN_SHARDS:
                old = next(iter(self._open))
                del self._open[old]
            self._open[shard] = safe_open(
                self.dir / shard, framework="pt", device="cpu")
        else:
            self._open[shard] = self._open.pop(shard)   # LRU refresh
        return self._open[shard]

    def get_shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._handle(name).get_slice(name).get_shape())

    def logical_shape(self, name: str) -> tuple[int, ...]:
        """Shape of the DECODED weight — what ``dequant_weight`` returns.

        Identical to the stored shape except for a declared MXFP4 nibble-pack,
        where two logical elements share each stored byte along the reduce dim
        (DSv4-Flash routed experts: stored ``[2048, 2048]`` I8 = logical
        ``[2048, 4096]``). The streaming plan sizes every output from metadata
        alone, so a physical shape here would size the CB payload at half its
        in_features and then fail the col_weights coverage gate."""
        shape = self.get_shape(name)
        if not shape:
            return shape
        from prismaquant.cb_source_decode import checkpoint_weight_to_live_name

        mxfp4 = getattr(self._fp8_scale_inv_map, "mxfp4_names", frozenset())
        if not mxfp4:
            return shape
        live = checkpoint_weight_to_live_name(name, profile=self._profile)
        if live in mxfp4:
            return (*shape[:-1], int(shape[-1]) * 2)
        return shape

    def get_dtype(self, name: str) -> torch.dtype:
        t = self._handle(name).get_slice(name)[0:0]
        return t.dtype

    def load(self, name: str) -> torch.Tensor:
        return self._handle(name).get_tensor(name)

    def _hdr(self, shard: Path) -> tuple[dict, int]:
        """Parsed safetensors header + data-start offset for one shard."""
        key = str(shard)
        if key not in self._shard_hdr:
            with open(shard, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(hlen))
            self._shard_hdr[key] = (hdr, 8 + hlen)
        return self._shard_hdr[key]

    def raw_slice(self, name: str) -> tuple[Path, int, int]:
        """``(shard_path, absolute file offset, nbytes)`` for a raw byte copy.

        The SOURCE-side twin of ``_PriorArtifact.raw_slice``, and the reason
        the native lane is a passthrough rather than a re-serialization: paired
        with ``_StreamWriter.add(copy_src=...)`` it moves a source tensor from
        the checkpoint's file to the artifact's file in 16 MiB chunks without
        ever constructing a torch.Tensor. On DSv4-Flash that is 147 GB of
        routed-expert payload the exporter never materialises — "stream, don't
        load" applied to the one lane where there is nothing to encode.
        """
        shard = self.dir / self.weight_map[name]
        hdr, data0 = self._hdr(shard)
        meta = hdr[name]
        lo, hi = meta["data_offsets"]
        return shard, data0 + int(lo), int(hi) - int(lo)

    def dequant_weight(self, weight_key: str) -> torch.Tensor:
        """Return the weight as fp32 for encoding. bf16/fp16 sources cast
        through the BF16 load contract; native-FP8 and declared MXFP4 sources
        use layer_streaming's profile-resolved scale/decode path (legacy
        ``weight_scale_inv`` and architecture-specific pairs such as DSv4's
        ``.scale`` siblings)."""
        from prismaquant.cb_source_decode import (
            cb_source_weight_bf16_value,
            checkpoint_weight_to_live_name,
        )

        w = self.load(weight_key)
        live_weight_name = checkpoint_weight_to_live_name(
            weight_key,
            profile=self._profile,
        )
        return cb_source_weight_bf16_value(
            w,
            model_weight_name=live_weight_name,
            fp8_scale_inv_map=self._fp8_scale_inv_map,
        )


# ---------------------------------------------------------------------------
# Streaming safetensors writer (analytic sizes -> header -> streamed data)
# ---------------------------------------------------------------------------

def _raw_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().flatten().view(torch.uint8).numpy().tobytes()


def _nbytes(dtype: torch.dtype, shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * torch.empty((), dtype=dtype).element_size()


_EXPORT_PIPELINE_ENV = "PRISMAQUANT_EXPORT_PIPELINE"
_EXPORT_PREFETCH_DEPTH_ENV = "PRISMAQUANT_EXPORT_PREFETCH_DEPTH"
_EXPORT_WRITE_QUEUE_BYTES_ENV = "PRISMAQUANT_EXPORT_WRITE_QUEUE_BYTES"
_SOURCE_MODEL_IDENTITY_CACHE_ENV = (
    "PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE"
)
_DSPARK_RENDER_RECIPE_SCHEMA = "prismaquant.dspark_cb_render_recipe.v1"
_DSPARK_RENDER_SOURCE_BINDING = "streamed_decoded_cb_source.v1"
_DEFAULT_EXPORT_PREFETCH_DEPTH = 1
_DEFAULT_EXPORT_WRITE_QUEUE_BYTES = 2 * (1 << 30)
_NO_SOURCE = object()


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _export_pipeline_enabled() -> bool:
    """Execution-strategy gate; deliberately absent from render identity."""

    return os.environ.get(_EXPORT_PIPELINE_ENV, "0") == "1"


def _source_model_identity_from_env(
    model_dir: str | Path,
) -> dict[str, object] | None:
    """Validate and compact the optional full-source export attestation."""
    cache_path = os.environ.get(_SOURCE_MODEL_IDENTITY_CACHE_ENV)
    if not cache_path:
        return None
    from prismaquant.cost_streaming import (
        validate_cached_streamed_model_identity,
    )

    identity = validate_cached_streamed_model_identity(
        model_dir, cache_path, require_complete_checkpoint=True
    )
    shards = identity.get("shards")
    checkpoint_weight_map = identity.get("checkpoint_weight_map")
    if not isinstance(shards, list) or not isinstance(
        checkpoint_weight_map, dict
    ):
        raise RuntimeError(
            f"{_SOURCE_MODEL_IDENTITY_CACHE_ENV} does not attest the complete "
            "indexed source checkpoint"
        )
    # The full object is several MiB for DSv4.  Its canonical content digest
    # already binds the normalized config, execution map, complete source
    # index, and every shard SHA; keep the artifact provenance compact while
    # retaining coverage counts that make accidental partial identities loud.
    return {
        "schema": identity.get("schema"),
        "content_sha256": identity.get("content_sha256"),
        "resolved_commit": identity.get("resolved_commit"),
        "checkpoint_shards": len(shards),
        "checkpoint_tensors": len(checkpoint_weight_map),
    }


def _require_production_source_model_identity(
    model_dir: str | Path,
    source_model_identity: dict[str, object] | None,
    *,
    allow_unstamped_research: bool,
) -> None:
    """Make the complete source attestation mandatory for DSv4 production.

    DeepSeek-V4's body runner deliberately does not materialize its MTP
    passthrough shards.  An optional identity cache therefore permits a
    superficially valid export whose provenance covers only the streamed body,
    not every byte copied into the artifact.  Synthetic/research renders retain
    their explicit escape hatch; the production DSv4 path must bind the full
    index-referenced checkpoint before it creates an output transaction.
    """
    if source_model_identity is not None or allow_unstamped_research:
        return
    config_path = Path(model_dir) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot determine whether production source identity is required: "
            f"{config_path}: {exc}"
        ) from exc
    model_type = config.get("model_type") if isinstance(config, dict) else None
    if str(model_type).lower().replace("-", "_") == "deepseek_v4":
        raise RuntimeError(
            "production DeepSeek-V4 streaming export requires "
            f"{_SOURCE_MODEL_IDENTITY_CACHE_ENV} bound to the complete "
            "index-referenced checkpoint (including passthrough/MTP shards)"
        )


def _bind_source_model_identity_provenance(
    quant_config: dict,
    source_model_identity: dict[str, object] | None,
) -> None:
    if source_model_identity is None:
        return
    provenance = quant_config.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(
            "streaming export cannot bind source identity without provenance"
        )
    if "source_model_identity" in provenance:
        raise RuntimeError(
            "streaming export quant config already carries a source identity"
        )
    provenance["source_model_identity"] = dict(source_model_identity)


def _canonical_json_digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_dspark_streaming_render_recipe(
    recipe: object,
    *,
    render_identity: object,
    source_model_identity: object,
    model_dir: Path,
    skeleton,
    assignment: Mapping[str, str],
) -> None:
    """Validate the immutable seed for one-pass DSpark source attestation."""

    required = {
        "schema",
        "source_binding",
        "source_model_identity",
        "source_config_sha256",
        "mtp_header_identity_sha256",
        "assignment_sha256",
        "col_weights_sha256",
        "render_identity_seed_sha256",
    }
    if not isinstance(recipe, Mapping) or set(recipe) != required:
        raise ValueError(
            "DSpark production render recipe stamp is missing or malformed"
        )
    if recipe.get("schema") != _DSPARK_RENDER_RECIPE_SCHEMA or recipe.get(
        "source_binding"
    ) != _DSPARK_RENDER_SOURCE_BINDING:
        raise ValueError("DSpark production render recipe schema is unsupported")
    if not isinstance(render_identity, Mapping):
        raise ValueError("DSpark production render recipe has no identity seed")
    if render_identity.get("source_weights_complete") is not False or (
        render_identity.get("source_weights_shapes")
        or render_identity.get("source_weights_content_sha256")
        or render_identity.get("source_weights_sha256") is not None
    ):
        raise ValueError(
            "DSpark one-pass render recipe must carry a pristine incomplete "
            "source identity seed"
        )
    if recipe.get("source_model_identity") != source_model_identity:
        raise ValueError(
            "DSpark render recipe source-checkpoint identity differs from "
            "the validated export source"
        )
    config_path = model_dir / "config.json"
    if recipe.get("source_config_sha256") != _file_sha256(config_path):
        raise ValueError("DSpark render recipe source config digest differs")
    header_identity = {
        name: {
            "dtype": str(skeleton.get_dtype(name)),
            "shape": list(skeleton.get_shape(name)),
        }
        for name in sorted(skeleton.keys())
        if str(name).startswith("mtp.")
    }
    if recipe.get("mtp_header_identity_sha256") != _canonical_json_digest(
        header_identity
    ):
        raise ValueError("DSpark render recipe MTP header identity differs")
    if recipe.get("assignment_sha256") != assignment_serialization_sha256(
        assignment
    ):
        raise ValueError("DSpark render recipe assignment digest differs")
    if recipe.get("col_weights_sha256") != render_identity.get(
        "col_weights_sha256"
    ):
        raise ValueError("DSpark render recipe imatrix digest differs")
    if recipe.get("render_identity_seed_sha256") != _canonical_json_digest(
        render_identity
    ):
        raise ValueError("DSpark render identity seed digest differs")


@dataclass(frozen=True)
class _StreamEntry:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    producer: object
    copy_src: object
    reader: object = None
    encoder: object = None

    @property
    def nbytes(self) -> int:
        return _nbytes(self.dtype, self.shape)


@dataclass(frozen=True)
class _CopyPayload:
    source: tuple[Path, int, int]


class _PipelineFailure:
    """First-error latch shared by all three pipeline stages."""

    def __init__(self):
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self.stop = threading.Event()

    def fail(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
        self.stop.set()

    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def raise_if_failed(self) -> None:
        error = self.error()
        if error is not None:
            raise error


class _ByteBudget:
    """Reserve encoded bytes before encode, bounding queued/in-flight output.

    A single tensor larger than the configured limit is admitted only while it
    owns the budget exclusively. This is necessary for real tensors larger
    than the default queue while still preventing multiple oversize outputs
    from accumulating.
    """

    def __init__(self, limit: int, failure: _PipelineFailure):
        self.limit = int(limit)
        self.failure = failure
        self.used = 0
        self._condition = threading.Condition()

    def acquire(self, amount: int) -> tuple[float, bool]:
        amount = int(amount)
        start = time.perf_counter()
        stalled = False
        with self._condition:
            if self.failure.stop.is_set():
                self.failure.raise_if_failed()
                raise RuntimeError("export pipeline stopped")
            while self.used and self.used + amount > self.limit:
                stalled = True
                if self.failure.stop.is_set():
                    self.failure.raise_if_failed()
                    raise RuntimeError("export pipeline stopped")
                self._condition.wait(timeout=0.05)
            self.used += amount
        return time.perf_counter() - start, stalled

    def release(self, amount: int) -> None:
        with self._condition:
            self.used -= int(amount)
            if self.used < 0:
                raise AssertionError("export pipeline byte budget underflow")
            self._condition.notify_all()

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()


class _OrderedResults:
    """Completion-order buffer whose consumer drains canonical entry order."""

    def __init__(self, failure: _PipelineFailure):
        self.failure = failure
        self._condition = threading.Condition()
        self._ready: dict[int, object] = {}

    def put(self, index: int, payload: object) -> None:
        with self._condition:
            self._ready[int(index)] = payload
            self._condition.notify_all()

    def get(self, index: int) -> tuple[object, float]:
        start = time.perf_counter()
        with self._condition:
            while index not in self._ready:
                if self.failure.stop.is_set():
                    self.failure.raise_if_failed()
                    raise RuntimeError("export pipeline stopped")
                self._condition.wait(timeout=0.05)
            return self._ready.pop(index), time.perf_counter() - start

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()


# On-disk dtypes each FormatSpec element/scale name can legally appear as.
# Sourced from the decode side (`layer_streaming._E8M0_SCALE_DTYPES` accepts
# all three E8M0 spellings) so a checkpoint the loader would decode is a
# checkpoint this exporter will copy, and no other.
_PASSTHROUGH_ELEMENT_DTYPES = {
    "fp8_e4m3": (torch.float8_e4m3fn,),
    "fp4_e2m1": (torch.int8, torch.uint8),
}
_PASSTHROUGH_SCALE_DTYPES = {
    "uint8_e8m0": (torch.float8_e8m0fnu, torch.uint8, torch.int8),
    "fp32": (torch.float32,),
}
# Logical elements per stored byte. >1 means a sub-byte pack, whose stored
# shape is NOT its logical shape.
_PASSTHROUGH_ELEMENTS_PER_BYTE = {"fp8_e4m3": 1, "fp4_e2m1": 2}


def _passthrough_scale_shape(spec, logical_shape) -> tuple[int, ...]:
    """The scale-plane shape one passthrough FormatSpec implies for a weight.

    Derived from the registry rather than hardcoded per format: a block format
    (``scale_block_shape``) tiles both dims, a group format divides the reduce
    dim by ``group_size``.  A future census entry gets its expectation for free.
    """
    rows, cols = int(logical_shape[0]), int(logical_shape[1])
    if spec.scale_block_shape is not None:
        block_rows, block_cols = (int(d) for d in spec.scale_block_shape)
        return (-(-rows // block_rows), -(-cols // block_cols))
    group = int(spec.group_size)
    if group <= 0 or cols % group:
        raise ValueError(
            f"{spec.name}: in_features={cols} is not a multiple of the "
            f"declared group size {group}")
    return (rows, cols // group)


def _assert_passthrough_planes_agree(qname, fmt, spec, wkey, wsh, wdtype,
                                     skey, ssh, sdtype) -> None:
    """Metadata-only check that a source pair really is this format on disk.

    The decode-side twins (`layer_streaming._check_mxfp4_packed_grid` /
    `_check_fp8_scale_grid`) need the tensors; this runs during PLANNING, from
    safetensors metadata alone, so a declaration/layout disagreement fails
    before 147 GB has been copied rather than after.
    """
    element_dtypes = _PASSTHROUGH_ELEMENT_DTYPES[str(spec.weight_element_dtype)]
    scale_dtypes = _PASSTHROUGH_SCALE_DTYPES[str(spec.scale_dtype_name)]
    per_byte = _PASSTHROUGH_ELEMENTS_PER_BYTE[str(spec.weight_element_dtype)]
    problems = []
    if len(wsh) != 2 or wdtype not in element_dtypes:
        problems.append(
            f"weight {wkey!r} is {wdtype}{wsh}, expected a 2-D "
            f"{'/'.join(str(d) for d in element_dtypes)} plane")
    if len(ssh) != 2 or sdtype not in scale_dtypes:
        problems.append(
            f"scale {skey!r} is {sdtype}{ssh}, expected a 2-D "
            f"{'/'.join(str(d) for d in scale_dtypes)} plane")
    if not problems:
        logical = (int(wsh[0]), int(wsh[1]) * per_byte)
        expected = _passthrough_scale_shape(spec, logical)
        if tuple(ssh) != expected:
            problems.append(
                f"scale {skey!r} is {tuple(ssh)}, but {spec.name} over a "
                f"logical {logical} weight implies {expected}")
    if problems:
        raise ValueError(
            f"{qname}: assigned {fmt}, but the checkpoint pair does not match "
            f"that on-disk contract — {'; '.join(problems)}. A passthrough "
            "copies bytes it does not interpret, so the source must already "
            "BE the format it is being shipped as.")


def _stock_output_specs(fmt: str, shape) -> list[tuple[str, torch.dtype, tuple]]:
    """Analytic on-disk ``(suffix, dtype, out_shape)`` list for a DENSE stock
    target whose source weight is ``shape`` (out=N, in=K). Mirrors
    ``export_native_compressed._quantize_2d`` output EXACTLY (verified against
    the packers) so the streaming header is sized without a pack:

      * NVFP4 (W4A4): ``weight_packed`` uint8 [N, K/2], ``weight_scale``
        fp8_e4m3 [N, K/16], ``weight_global_scale`` fp32 [1],
        ``input_global_scale`` fp32 [1].
      * FP8_E4M3 (W8A8 per-channel): ``weight`` fp8_e4m3 [N, K],
        ``weight_scale`` fp32 [N, 1].
    """
    n, k = int(shape[-2]), int(shape[-1])
    if fmt == "NVFP4":
        return [
            ("weight_packed", torch.uint8, (n, k // 2)),
            ("weight_scale", torch.float8_e4m3fn, (n, k // 16)),
            ("weight_global_scale", torch.float32, (1,)),
            ("input_global_scale", torch.float32, (1,)),
        ]
    if fmt == "FP8_E4M3":
        return [
            ("weight", torch.float8_e4m3fn, (n, k)),
            ("weight_scale", torch.float32, (n, 1)),
        ]
    raise ValueError(f"no stock streaming spec for {fmt!r}")


def _requant_output_specs(fmt: str, shape) -> list[tuple[str, torch.dtype, tuple]]:
    """On-disk ``(suffix, dtype, out_shape)`` for a re-quantized native rung.

    Deliberately the SAME ``weight`` / ``weight_scale`` suffix pair the
    FP8_E4M3 stock rung and the CT-normalized FP8_SOURCE lane already use: the
    suffix names what a plane IS (elements, and their scales), not which codec
    produced it, so a new native rung does not invent a third spelling for the
    same two roles.

      * MXFP8_UE8M0_G32: ``weight`` fp8_e4m3 [N, K], ``weight_scale``
        float8_e8m0fnu [N, K/32]. The scale plane is the NATIVE E8M0 dtype, not
        the ``uint8`` the compressed-tensors MXFP8 scheme serializes — that is
        one of the reasons this rung is its own format rather than the stock
        one (see the FormatSpec comment in format_registry).
    """
    n, k = int(shape[-2]), int(shape[-1])
    spec = _fr_get_format(fmt)
    group = int(spec.group_size)
    if group <= 0 or k % group:
        raise ValueError(
            f"{fmt}: in_features={k} is not a multiple of the declared group "
            f"size {group}; check_format_applicability should have masked this "
            "target before it reached the exporter")
    return [
        ("weight", torch.float8_e4m3fn, (n, k)),
        ("weight_scale", torch.float8_e8m0fnu, (n, k // group)),
    ]


def _requant_pack(fmt: str, w: torch.Tensor) -> dict[str, torch.Tensor]:
    """Encode one dense weight into its re-quantized on-disk planes.

    Routes through the SAME registry codec the cost stage measured
    (``mx_formats.mxfp8_ue8m0_qdq``), so priced error and shipped bytes are one
    rendering rather than two implementations that agree by inspection.
    """
    canon = canonical_format_name(fmt)
    if canon == "MXFP8_UE8M0_G32":
        from prismaquant.mx_formats import mxfp8_ue8m0_qdq

        spec = _fr_get_format(canon)
        result = mxfp8_ue8m0_qdq(
            w.to(torch.float32), group_size=int(spec.group_size)
        )
        return {"weight": result.quant, "weight_scale": result.scale}
    raise ValueError(f"no re-quant streaming packer for {fmt!r}")


def _merge_pipeline_timings(
    total: dict[str, float] | None, shard: dict[str, float],
) -> dict[str, float]:
    """Sum the per-shard pipeline timings into one report for the whole write."""
    if total is None:
        return dict(shard)
    for key, value in shard.items():
        total[key] = total.get(key, 0.0) + value
    return total


class _StreamWriter:
    """Two-pass safetensors writer. ``add`` records (name, dtype, shape) and a
    zero-arg ``producer`` that yields the tensor at write time; ``write`` lays
    out contiguous offsets, writes the header, then streams every producer's
    bytes in order — one output tensor resident at a time.

    With ``shard_bytes`` the same stream is published as the HF-standard shard
    layout (:mod:`prismaquant.shard_layout`) instead of one container: the
    entry sequence is partitioned up front, so every shard's name and header
    are known before a byte is written and the layout is a pure function of
    the emit order and the budget."""

    def __init__(self):
        self._entries: list[_StreamEntry] = []
        # Set only after an atomic publication succeeds.  The production
        # exporter feeds this digest into the shipcard's immutable weight
        # manifest, avoiding a second sequential read of the finished
        # 100GB-class DSv4 container.  The scalar pair describes the single
        # published container and stays ``None`` on a sharded publication;
        # ``last_weight_manifest_files`` is the general answer and is always
        # populated on success.
        self.last_content_sha256: str | None = None
        self.last_content_bytes: int | None = None
        self.last_weight_manifest_files: dict[str, dict[str, object]] = {}
        # Layout-INVARIANT payload identity: sha256 of each tensor's raw bytes,
        # hashed in the same single pass that hashes the containers.  Two
        # exports of the same tensors at different shard budgets agree here and
        # differ in `model_sha` (which binds container filenames and sizes,
        # `shipcard.compute_model_sha`), so a reshard remains recognisable as
        # the same model without changing what identity means.
        self.last_tensor_content_sha256: dict[str, str] = {}

    def add(self, name, dtype, shape, producer, copy_src=None, *, reader=None,
            encoder=None):
        """Record an output tensor. ``producer`` yields it at write time; when
        ``copy_src=(path, file_offset, nbytes)`` is given (DELTA-EXPORT reuse)
        those raw bytes are streamed straight from a prior artifact's shard file
        instead — ``producer`` is then unused (may be None).

        ``reader`` + ``encoder`` are an optional execution-only split of the
        same producer. The serial path ignores them and calls ``producer``
        verbatim; the flag-gated pipeline reads ahead with ``reader`` and feeds
        its value to ``encoder``. Both must be supplied together.
        """
        if (reader is None) != (encoder is None):
            raise ValueError(f"{name}: reader and encoder must be paired")
        if copy_src is not None and (reader is not None or encoder is not None):
            raise ValueError(f"{name}: raw-copy and read/encode paths conflict")
        self._entries.append(_StreamEntry(
            str(name), dtype, tuple(int(d) for d in shape), producer, copy_src,
            reader, encoder,
        ))

    def names(self) -> list[str]:
        return [entry.name for entry in self._entries]

    def _plan_containers(
        self, path: Path, shard_bytes: int | None,
    ) -> list[tuple[str, list[_StreamEntry]]]:
        """``[(filename, entries)]`` in emit order, before any byte is written.

        The duplicate-name refusal runs here so it precedes every filesystem
        effect: a name planned twice keeps one header span while both blobs are
        still streamed, silently corrupting every offset after it.
        """
        seen: set[str] = set()
        for entry in self._entries:
            if entry.name in seen:
                raise AssertionError(
                    f"{entry.name}: planned twice; two emit paths claim the "
                    "same output tensor")
            seen.add(entry.name)
        if not self._entries:
            raise AssertionError("streaming export planned no output tensors")

        if shard_bytes is None:
            return [(path.name, list(self._entries))]

        groups = plan_shards(
            [(entry.name, entry.nbytes) for entry in self._entries],
            shard_bytes,
        )
        if len(groups) == 1:
            return [(path.name, list(self._entries))]
        if path.name != SINGLE_CONTAINER_NAME:
            # The shard filenames are derived from the standard layout, not
            # from `path`, so a caller sharding under some other basename would
            # publish files no loader associates with the name it asked for.
            raise ValueError(
                f"{path}: a sharded publication is named by "
                f"{SINGLE_CONTAINER_NAME!r}; pass that path or raise "
                "shard_bytes above the artifact size"
            )
        names = container_names(len(groups))
        planned: list[tuple[str, list[_StreamEntry]]] = []
        cursor = 0
        for name, group in zip(names, groups):
            planned.append((name, self._entries[cursor:cursor + len(group)]))
            cursor += len(group)
        assert cursor == len(self._entries)
        return planned

    @staticmethod
    def _shard_header(entries: Sequence[_StreamEntry]) -> tuple[bytes, int]:
        """``(header bytes, payload bytes)`` for one container's entries."""
        header: dict[str, dict] = {}
        off = 0
        for entry in entries:
            nb = entry.nbytes
            header[entry.name] = {
                "dtype": _ST_DTYPE[entry.dtype],
                "shape": list(entry.shape),
                "data_offsets": [off, off + nb],
            }
            off += nb
        header["__metadata__"] = {"format": "pt", "quant_method": "gridbook"}
        return json.dumps(header, separators=(",", ":")).encode("utf-8"), off

    def write(self, path: Path, *, shard_bytes: int | None = None,
              before_publish=None,
              _pipeline_encode_workers: int = 1) -> dict[str, float] | None:
        """Publish the recorded stream atomically.

        ``shard_bytes=None`` writes exactly one container at ``path``.  An
        integer budget publishes the HF-standard layout: one shard keeps
        ``path``; N > 1 becomes ``model-XXXXX-of-YYYYY.safetensors`` plus
        ``model.safetensors.index.json``.  Every container is streamed to a
        temporary file first and ``before_publish`` runs once, after the last
        producer, so an abort at any point leaves no partial artifact.
        """
        self.last_content_sha256 = None
        self.last_content_bytes = None
        self.last_weight_manifest_files = {}
        self.last_tensor_content_sha256 = {}
        path = Path(path)
        out_dir = path.parent
        planned = self._plan_containers(path, shard_bytes)
        sharded = len(planned) > 1

        # A safetensors header binds names/dtypes/shapes, not the source
        # weights, imatrix, codebooks, or exporter implementation.  Reusing a
        # same-shaped partial file can therefore preserve bytes produced by a
        # different render while every final span/size assertion still passes.
        # Resume stays disabled until the header carries one immutable digest
        # covering all of those producer inputs.
        reserved = [out_dir / name for name, _ in planned]
        if sharded:
            reserved.append(out_dir / SHARD_INDEX_NAME)
            # A stale run at a different shard COUNT leaves containers this
            # plan never names; they would be indistinguishable from this
            # export's own output to any consumer that globs *.safetensors.
            reserved.extend(
                sibling for sibling in sorted(out_dir.glob("*.safetensors"))
                if SHARD_NAME_RE.fullmatch(sibling.name) is not None
            )
        for candidate in reserved:
            if os.path.lexists(candidate):
                raise RuntimeError(
                    f"{candidate}: refusing an unbound streaming resume. The "
                    "existing file header does not prove source/imatrix/"
                    "codebook/exporter identity; use a fresh output directory."
                )
        temps = [out_dir / f".{name}.tmp" for name, _ in planned]
        for temp_path in temps:
            if os.path.lexists(temp_path):
                raise RuntimeError(
                    f"{temp_path}: refusing to overwrite a stale or aliased "
                    "streaming-export temporary file"
                )

        cuda = torch.cuda.is_available()
        owned_temps: list[Path] = []
        try:
            tensor_digests: dict[str, str] = {}
            manifest: dict[str, dict[str, object]] = {}
            timings: dict[str, float] | None = None
            index_offset = 0
            for (name, entries), temp_path in zip(planned, temps):
                hjson, payload_bytes = self._shard_header(entries)
                data0 = 8 + len(hjson)
                digest = hashlib.sha256()
                bytes_written = 0

                class _HashingWriter:
                    """Writer that hashes exactly the bytes it publishes.

                    ``begin_tensor``/``end_tensor`` bracket one tensor's span so
                    the layout-invariant payload digest costs no extra read --
                    the same bytes feed both the container digest and the
                    per-tensor one.
                    """

                    def __init__(self) -> None:
                        self._tensor: str | None = None
                        self._tensor_digest = None

                    def begin_tensor(self, tensor_name: str) -> None:
                        self._tensor = tensor_name
                        self._tensor_digest = hashlib.sha256()

                    def end_tensor(self) -> None:
                        if self._tensor is not None:
                            tensor_digests[self._tensor] = (
                                self._tensor_digest.hexdigest()
                            )
                        self._tensor = None
                        self._tensor_digest = None

                    def write(self, payload) -> int:
                        nonlocal bytes_written
                        view = memoryview(payload)
                        written = raw_file.write(view)
                        if written != len(view):
                            raise OSError(
                                "short write while streaming safetensors: "
                                f"{written} of {len(view)} bytes"
                            )
                        digest.update(view)
                        if self._tensor_digest is not None:
                            self._tensor_digest.update(view)
                        bytes_written += written
                        return written

                with open(temp_path, "xb") as raw_file:
                    owned_temps.append(temp_path)
                    f = _HashingWriter()
                    f.write(struct.pack("<Q", len(hjson)))
                    f.write(hjson)
                    if _export_pipeline_enabled():
                        shard_timings = self._write_pipeline(
                            f,
                            entries,
                            cuda=cuda,
                            encode_workers=int(_pipeline_encode_workers),
                            index_offset=index_offset,
                        )
                        timings = _merge_pipeline_timings(
                            timings, shard_timings)
                    else:
                        self._write_serial(
                            f, entries, cuda=cuda, index_offset=index_offset)
                expected_bytes = data0 + payload_bytes
                if bytes_written != expected_bytes:
                    raise AssertionError(
                        "streamed safetensors byte count differs from its "
                        f"header: {bytes_written} != {expected_bytes}"
                    )
                manifest[name] = {
                    "bytes": bytes_written,
                    "sha256": digest.hexdigest(),
                }
                index_offset += len(entries)

            if set(tensor_digests) != {e.name for e in self._entries}:
                raise AssertionError(
                    "streaming writer did not attest every published tensor: "
                    f"{len(tensor_digests)} of {len(self._entries)}"
                )
            if before_publish is not None:
                before_publish()
            for (name, _entries), temp_path in zip(planned, temps):
                os.replace(temp_path, out_dir / name)
                owned_temps.remove(temp_path)
            if sharded:
                weight_map = {
                    entry.name: name
                    for name, entries in planned for entry in entries
                }
                write_shard_index(
                    out_dir, weight_map,
                    sum(entry.nbytes for entry in self._entries),
                )
                print(
                    f"[export-cb-stream] published {len(planned)} safetensors "
                    f"shard(s) + {SHARD_INDEX_NAME}", flush=True)
            else:
                only = planned[0][0]
                self.last_content_sha256 = str(manifest[only]["sha256"])
                self.last_content_bytes = int(manifest[only]["bytes"])
            self.last_weight_manifest_files = manifest
            self.last_tensor_content_sha256 = tensor_digests
            return timings
        except BaseException:
            for temp_path in owned_temps:
                if os.path.lexists(temp_path):
                    temp_path.unlink()
            raise

    def _write_serial(self, f, entries: Sequence[_StreamEntry], *, cuda: bool,
                      index_offset: int = 0) -> None:
        """The pre-pipeline loop, kept as the exact flag-off implementation."""

        for i, entry in enumerate(entries, start=index_offset):
            name, dtype, shape = entry.name, entry.dtype, entry.shape
            f.begin_tensor(name)
            try:
                if entry.copy_src is not None:
                    self._write_copy(f, i, entry)
                    continue
                t = entry.producer()
                if t.dtype != dtype or tuple(t.shape) != shape:
                    raise AssertionError(
                        f"{name}: produced {t.dtype}{tuple(t.shape)} != "
                        f"declared {dtype}{shape}")
                b = _raw_bytes(t)
                if len(b) != entry.nbytes:
                    raise AssertionError(f"{name}: byte count mismatch")
                f.write(b)
                del t, b
            finally:
                f.end_tensor()
            self._cuda_hygiene(i, entry, cuda=cuda)

    def _write_copy(self, f, index: int, entry: _StreamEntry) -> None:
        # DELTA-EXPORT/source passthrough: sequential large reads from a prior
        # artifact or source checkpoint. Chunking bounds the writer working set.
        src_path, foff, nb = entry.copy_src
        if nb != entry.nbytes:
            raise AssertionError(
                f"{entry.name}: copy_src {nb}B != declared {entry.nbytes}B")
        with open(src_path, "rb") as sf:
            sf.seek(foff)
            remaining = nb
            while remaining:
                chunk = sf.read(min(remaining, 1 << 24))
                if not chunk:
                    raise AssertionError(
                        f"{entry.name}: prior artifact truncated at offset "
                        f"{foff} (needed {nb}B)")
                f.write(chunk)
                remaining -= len(chunk)
        if index % 50 == 0 or nb > (1 << 30):
            print(f"[export-cb-stream] {index + 1}/"
                  f"{len(self._entries)} {entry.name} copied "
                  f"{nb / 2**30:.2f}G from prior", flush=True)

    def _cuda_hygiene(self, index: int, entry: _StreamEntry, *, cuda: bool) -> None:
        if not cuda:
            return
        # Unified-memory hygiene: differently-shaped 10GB-class pack
        # transients must not accumulate as cached segments.
        torch.cuda.empty_cache()
        if index % 20 == 0 or entry.nbytes > (1 << 30):
            print(f"[export-cb-stream] {index + 1}/"
                  f"{len(self._entries)} {entry.name} cuda alloc "
                  f"{torch.cuda.memory_allocated() / 2**30:.1f}G reserved "
                  f"{torch.cuda.memory_reserved() / 2**30:.1f}G", flush=True)

    def _write_pipeline(self, f, entries: Sequence[_StreamEntry], *,
                        cuda: bool, encode_workers: int = 1,
                        index_offset: int = 0) -> dict[str, float]:
        """Read -> encode -> bounded ordered-write execution strategy.

        Production uses one encode worker, preserving the existing ordered
        encode stream. ``encode_workers`` exists only as a test seam that can
        force completion reordering and prove the writer's ordering contract.
        """

        if encode_workers <= 0:
            raise ValueError("pipeline encode_workers must be positive")
        depth = _positive_env_int(
            _EXPORT_PREFETCH_DEPTH_ENV, _DEFAULT_EXPORT_PREFETCH_DEPTH)
        write_bytes = _positive_env_int(
            _EXPORT_WRITE_QUEUE_BYTES_ENV,
            _DEFAULT_EXPORT_WRITE_QUEUE_BYTES,
        )
        failure = _PipelineFailure()
        budget = _ByteBudget(write_bytes, failure)
        results = _OrderedResults(failure)
        read_queue: queue.Queue = queue.Queue()
        prefetch_slots = threading.BoundedSemaphore(depth)
        sentinel = object()
        timings = {
            "read_busy": 0.0,
            "read_stall": 0.0,
            "encode_busy": 0.0,
            "encode_stall": 0.0,
            "write_busy": 0.0,
            "write_stall": 0.0,
            "backpressure_stalls": 0,
        }
        timing_lock = threading.Lock()
        wall_start = time.perf_counter()

        def add_timing(key: str, value: float) -> None:
            with timing_lock:
                timings[key] += value

        def queue_put(item) -> float:
            start = time.perf_counter()
            read_queue.put(item)
            return time.perf_counter() - start

        def acquire_prefetch_slot() -> float:
            start = time.perf_counter()
            while not failure.stop.is_set():
                if prefetch_slots.acquire(timeout=0.05):
                    return time.perf_counter() - start
            failure.raise_if_failed()
            raise RuntimeError("export pipeline stopped")

        def read_stage() -> None:
            try:
                for index, entry in enumerate(entries):
                    if failure.stop.is_set():
                        break
                    owns_prefetch_slot = False
                    if entry.copy_src is not None or entry.reader is None:
                        started = time.perf_counter()
                        source = _NO_SOURCE
                    else:
                        add_timing("read_stall", acquire_prefetch_slot())
                        owns_prefetch_slot = True
                        started = time.perf_counter()
                        try:
                            source = entry.reader()
                        except BaseException:
                            prefetch_slots.release()
                            raise
                    add_timing("read_busy", time.perf_counter() - started)
                    add_timing("read_stall", queue_put(
                        (index, entry, source, owns_prefetch_slot)))
            except BaseException as exc:
                failure.fail(exc)
                budget.wake()
                results.wake()
            finally:
                try:
                    queue_put(sentinel)
                except BaseException:
                    pass

        def encode_one(index: int, entry: _StreamEntry, source: object) -> None:
            try:
                started = time.perf_counter()
                if entry.copy_src is not None:
                    payload = _CopyPayload(entry.copy_src)
                else:
                    t = (entry.encoder(source) if entry.encoder is not None
                         else entry.producer())
                    if t.dtype != entry.dtype or tuple(t.shape) != entry.shape:
                        raise AssertionError(
                            f"{entry.name}: produced {t.dtype}{tuple(t.shape)} "
                            f"!= declared {entry.dtype}{entry.shape}")
                    payload = _raw_bytes(t)
                    if len(payload) != entry.nbytes:
                        raise AssertionError(
                            f"{entry.name}: byte count mismatch")
                    del t
                add_timing("encode_busy", time.perf_counter() - started)
                results.put(index, payload)
            except BaseException as exc:
                budget.release(entry.nbytes)
                failure.fail(exc)
                budget.wake()
                results.wake()

        def write_stage() -> None:
            try:
                for index, entry in enumerate(entries):
                    payload, stalled = results.get(index)
                    add_timing("write_stall", stalled)
                    started = time.perf_counter()
                    f.begin_tensor(entry.name)
                    try:
                        if isinstance(payload, _CopyPayload):
                            self._write_copy(
                                f, index + index_offset, entry)
                        else:
                            f.write(payload)
                    finally:
                        f.end_tensor()
                    add_timing("write_busy", time.perf_counter() - started)
                    budget.release(entry.nbytes)
                    self._cuda_hygiene(
                        index + index_offset, entry, cuda=cuda)
            except BaseException as exc:
                failure.fail(exc)
                budget.wake()
                results.wake()

        reader = threading.Thread(
            target=read_stage, name="pq-export-reader", daemon=True)
        writer = threading.Thread(
            target=write_stage, name="pq-export-writer", daemon=True)
        reader.start()
        writer.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=encode_workers,
                    thread_name_prefix="pq-export-encode") as pool:
                pending: set[concurrent.futures.Future] = set()
                while True:
                    if len(pending) >= encode_workers:
                        done, pending = concurrent.futures.wait(
                            pending,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            future.result()
                        failure.raise_if_failed()
                    started = time.perf_counter()
                    try:
                        item = read_queue.get(timeout=0.05)
                    except queue.Empty:
                        add_timing("encode_stall", time.perf_counter() - started)
                        failure.raise_if_failed()
                        continue
                    add_timing("encode_stall", time.perf_counter() - started)
                    failure.raise_if_failed()
                    if item is sentinel:
                        break
                    index, entry, source, owns_prefetch_slot = item
                    if owns_prefetch_slot:
                        prefetch_slots.release()
                    stall, did_stall = budget.acquire(entry.nbytes)
                    add_timing("encode_stall", stall)
                    if did_stall:
                        add_timing("backpressure_stalls", 1)
                    try:
                        pending.add(pool.submit(
                            encode_one, index, entry, source))
                    except BaseException:
                        budget.release(entry.nbytes)
                        raise
                for future in pending:
                    future.result()
        except BaseException as exc:
            failure.fail(exc)
            budget.wake()
            results.wake()
        finally:
            reader.join()
            writer.join()
        failure.raise_if_failed()
        timings["wall"] = time.perf_counter() - wall_start
        print(
            "[export-cb-stream] pipeline timings "
            f"wall={timings['wall']:.3f}s "
            f"read_busy={timings['read_busy']:.3f}s "
            f"read_stall={timings['read_stall']:.3f}s "
            f"encode_busy={timings['encode_busy']:.3f}s "
            f"encode_stall={timings['encode_stall']:.3f}s "
            f"write_busy={timings['write_busy']:.3f}s "
            f"write_stall={timings['write_stall']:.3f}s "
            f"backpressure_stalls={int(timings['backpressure_stalls'])} "
            f"prefetch_depth={depth} write_queue_bytes={write_bytes}",
            flush=True,
        )
        return timings


# ---------------------------------------------------------------------------
# Per-expert -> stacked plan
# ---------------------------------------------------------------------------

def _packed_expert_param_names(profile) -> frozenset[str]:
    if profile is not None:
        try:
            names = frozenset(profile.packed_expert_param_names())
            if names:
                return names
        except Exception:
            pass
    return frozenset(_LEGACY_PACKED_PROJECTIONS)


def _packed_expert_projection_names(profile, packed_proj: str) -> tuple[str, ...]:
    if profile is not None:
        try:
            names = tuple(profile.packed_expert_projection_names(packed_proj))
            if names:
                return names
        except Exception:
            pass
    return _LEGACY_PACKED_PROJECTIONS.get(packed_proj, (packed_proj,))


def _plan_expert_stacks(skeleton: _LazySkeleton, profile=None) -> dict[str, dict]:
    """Group per-expert on-disk tensors into stacked-output plans keyed by the
    LIVE packed qname (``…experts.gate_up_proj`` = fused gate+up, or
    ``…experts.down_proj``). Projection names and fusion order come from the
    model profile (for example LFM uses ``w1/w3`` + ``w2`` rather than
    ``gate_proj/up_proj`` + ``down_proj``). The legacy projection names remain
    as a fallback for profile-less synthetic checkpoints.

    Returns ``{checkpoint_experts_prefix: {projection: {expert_id: base}}}``.
    """
    experts: dict[str, dict[str, dict[int, str]]] = {}
    regex = None
    if profile is not None:
        try:
            regex = profile.per_expert_moe_regex()
        except Exception:
            regex = None
    pat = None
    if regex:
        pat = re.compile(regex[len("re:"):] if regex.startswith("re:")
                         else regex)

    def _matching_name(name: str, base: str) -> str | None:
        """The spelling of this per-expert tensor that the profile's
        per-expert regex matches, or None.

        Three spellings are tried, in increasing distance from the bytes on
        disk: the checkpoint base itself, its vLLM-internal name, and its LIVE
        (transformers) name via ``checkpoint_to_live_name``. The live bridge is
        what admits a checkpoint whose per-expert naming is its OWN
        (DSv4-Flash stores ``layers.N.ffn.experts.{i}.w{1,2,3}`` while the
        recipe, the imatrix, the probe and the regex all speak
        ``model.layers.N.mlp.experts.{i}.{gate,up,down}_proj``). Whichever
        spelling matched becomes the group key, so ``_resolve_target`` finds
        the group under the recipe name while the members stay CHECKPOINT
        bases for ``_expert_weight`` to read."""
        if pat is None:
            return None
        candidates = [base]
        for resolve in (
            lambda: profile.to_vllm_internal_name(base),
            lambda: _checkpoint_to_live_base(name, profile),
        ):
            try:
                cand = resolve()
            except Exception:
                cand = None
            if cand and cand not in candidates:
                candidates.append(cand)
        for cand in candidates:
            if pat.match(cand):
                return cand
        return None

    for name in skeleton.keys():
        if not name.endswith(".weight"):
            continue
        base = name[:-len(".weight")]
        if pat is not None:
            matched = _matching_name(name, base)
            if matched is None:
                continue
            try:
                head, proj = matched.rsplit(".", 1)
                prefix, idx_s = head.rsplit(".", 1)
            except ValueError:
                continue
            if not idx_s.isdigit():
                continue
            try:
                parent = profile.packed_expert_parent_for_projection(proj)
            except Exception:
                parent = None
            if parent is None:
                continue
            idx = int(idx_s)
        else:
            m = _LEGACY_EXPERT_RE.match(name)
            if not m:
                continue
            prefix, idx, proj = m.group(1), int(m.group(2)), m.group(3)
        experts.setdefault(prefix, {}).setdefault(proj, {})[idx] = base
    return experts


_DSPARK_CB_EXPERT_WEIGHT_RE = re.compile(
    r"^(mtp[.](?P<stage>\d+)[.]ffn[.]experts)[.]"
    r"(?P<expert>\d+)[.](?P<projection>w1|w2|w3)[.]weight$"
)
_DSPARK_CB_EXPERT_RECIPE_PROJECTION = {
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}


def _plan_dspark_cb_expert_stacks(
    skeleton: _LazySkeleton,
    source_config: dict,
) -> dict[str, dict[str, dict[int, str]]]:
    """Plan physical ``mtp.*`` experts for a Gridbook DSpark sidecar.

    The ordinary DeepSeek profile intentionally drops MTP from its body/probe
    namespace, so the generic expert planner cannot see these tensors.  This
    adapter keeps the allocator/encoder vocabulary (``gate/up/down_proj``)
    while binding each member to the released checkpoint's physical
    ``w1/w3/w2`` base.  The closed DSpark source-layout validator runs before
    this helper; the assertions here protect the exporter-specific grouping
    from ever becoming partial or reordered.
    """

    try:
        n_experts = int(source_config["n_routed_experts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "DSpark CB sidecar requires integer n_routed_experts"
        ) from exc
    if n_experts <= 0:
        raise ValueError(
            "DSpark CB sidecar requires positive n_routed_experts"
        )

    groups: dict[str, dict[str, dict[int, str]]] = {}
    for name in skeleton.keys():
        match = _DSPARK_CB_EXPERT_WEIGHT_RE.fullmatch(str(name))
        if match is None:
            continue
        prefix = match.group(1)
        expert_id = int(match.group("expert"))
        source_projection = match.group("projection")
        recipe_projection = _DSPARK_CB_EXPERT_RECIPE_PROJECTION[
            source_projection
        ]
        recipe_target = f"{prefix}.{expert_id}.{recipe_projection}"
        physical_base = str(name)[: -len(".weight")]
        resolved = dspark_cb_physical_source_for_recipe_target(
            recipe_target, source_config
        )
        if resolved != physical_base:
            raise AssertionError(
                f"{recipe_target}: DSpark source planner resolved {resolved!r} "
                f"instead of checkpoint base {physical_base!r}"
            )
        projection_members = groups.setdefault(prefix, {}).setdefault(
            recipe_projection, {}
        )
        if expert_id in projection_members:
            raise ValueError(
                f"{recipe_target}: duplicate DSpark expert source member"
            )
        projection_members[expert_id] = physical_base

    expected_stages = {0, 1, 2}
    observed_stages = {
        int(prefix.split(".", 2)[1]) for prefix in groups
    }
    if observed_stages != expected_stages:
        raise ValueError(
            "DSpark CB expert planner requires exactly physical stages "
            f"{sorted(expected_stages)}, got {sorted(observed_stages)}"
        )
    expected_ids = list(range(n_experts))
    expected_projections = set(_DSPARK_CB_EXPERT_RECIPE_PROJECTION.values())
    for prefix, projections in sorted(groups.items()):
        if set(projections) != expected_projections:
            raise ValueError(
                f"{prefix}: DSpark CB expert projections must be "
                f"{sorted(expected_projections)}, got {sorted(projections)}"
            )
        for projection, members in sorted(projections.items()):
            if sorted(members) != expected_ids:
                raise ValueError(
                    f"{prefix}.{projection}: DSpark CB expert ids must be "
                    f"contiguous 0..{n_experts - 1}, got "
                    f"{sorted(members)[:8]}"
                )
    return groups


def _checkpoint_to_live_base(weight_key: str, profile) -> str | None:
    """Checkpoint ``<base>.weight`` key -> live module base, or None."""
    if profile is None:
        return None
    live = profile.checkpoint_to_live_name(weight_key, multimodal=False)
    if not live or not live.endswith(".weight"):
        return None
    return live[: -len(".weight")]


def _expert_member_qnames(prefix, packed_proj, members, profile
                          ) -> dict[tuple[str, int], str]:
    """``{(projection, expert_id): recipe qname}`` for one packed stack, in the
    profile's fusion order. The recipe qname is rebuilt from the GROUP KEY, so
    it is exactly the name the allocator, the imatrix and the render identity
    carry for that per-expert Linear."""
    projections = _packed_expert_projection_names(profile, packed_proj)
    n = _n_experts(members, projections)
    return {(proj, e): f"{prefix}.{e}.{proj}"
            for proj in projections
            for e in range(n)}


def _collapse_per_expert_assignment(assignment, expert_groups, profile):
    """Collapse an EXPANDED per-expert assignment into packed-stack targets.

    The allocator decides an expert group ATOMICALLY but writes its
    layer_config EXPANDED per tensor (allocator.py:4662-4668), so one DSv4
    layer arrives as 768 entries (256 experts x gate/up/down) that all agree on
    one format. Gridbook's stacked-expert contract has no per-expert spelling at
    all: its loader anchors on ``.experts.{gate_up_proj,down_proj}.<leaf>``
    (gridbook moe_toplevel_loader.py:125-132), and a per-expert CB key misses
    every resolver silently and then dies as a bare ``KeyError`` in the
    architecture's own loader. So the export must name the two stacks per layer
    instead — the reduction the allocator never had to perform.

    A stack whose members do NOT all carry the same format is REFUSED rather
    than named: "All experts of one stack MUST share one format and one
    codebook ... experts MAY differ across layers but MUST be uniform within a
    layer" (gridbook docs/SPEC.md:288-291). The runtime's own uniformity checks
    are a byte-width assert and a scheme-signature comparison, so a mixed stack
    that reached serving would be a load-time crash at best.

    A SOURCE-PASSTHROUGH group (``cb_export_config
    .SOURCE_PASSTHROUGH_EXPORT_FORMATS``) is deliberately NOT collapsed. The
    collapse exists to name a packed parent that gridbook's CB loader anchors
    on; a passthrough group has no CB loader and no packed parent, because the
    exporter copies the per-expert checkpoint tensors and whichever loader owns
    them reads them under their own names. Naming ``…experts.gate_up_proj`` for
    it would promise a stack that is never written — the same reason
    export_nvfp4_cb excludes FP8_SOURCE from ``_pack_skeleton_experts``. The
    uniformity check still runs first, so a layer that MIXES a CB rung with a
    passthrough is refused rather than silently split across two loaders.

    Returns ``(collapsed_assignment, members_by_target, report)``. Members are
    ``{packed_qname: {(projection, expert_id): member_recipe_qname}}`` so the
    imatrix, the render identity and the source-value verification all stay
    keyed by the names the recipe actually carries.
    """
    collapsed = dict(assignment)
    members_by_target: dict[str, dict[tuple[str, int], str]] = {}
    report: dict[str, object] = {"stacks": 0, "members": 0}
    packed_names = _packed_expert_param_names(profile)
    for prefix in sorted(expert_groups):
        group = expert_groups[prefix]
        consumed: set[str] = set()
        # Widest parent first: the profile-less fallback vocabulary lists
        # `gate_proj`/`up_proj` as packed parents alongside `gate_up_proj`, and
        # a projection must land in the FUSED stack, not in a singleton one.
        for packed_proj in sorted(
            packed_names,
            key=lambda name: (
                -len(_packed_expert_projection_names(profile, name)), name),
        ):
            projections = _packed_expert_projection_names(profile, packed_proj)
            if any(p in consumed for p in projections):
                continue
            if any(p not in group for p in projections):
                continue
            try:
                member_qnames = _expert_member_qnames(
                    prefix, packed_proj, group, profile)
            except ValueError:
                continue
            present = {q for q in member_qnames.values() if q in assignment}
            if not present:
                continue
            packed_qname = f"{prefix}.{packed_proj}"
            if packed_qname in assignment:
                raise ValueError(
                    f"{packed_qname}: the layer config carries BOTH the packed "
                    f"stack and {len(present)} per-expert member(s) for it; "
                    "one allocation must describe each serving unit once")
            missing = sorted(set(member_qnames.values()) - present)
            if missing:
                raise ValueError(
                    f"{packed_qname}: {len(present)} of "
                    f"{len(member_qnames)} per-expert members are in the layer "
                    f"config; a packed stack is exported whole or not at all "
                    f"(missing e.g. {missing[:4]})")
            formats = sorted({str(assignment[q]) for q in present})
            if len(formats) > 1:
                by_fmt = {
                    fmt: sorted(q for q in present
                                if str(assignment[q]) == fmt)[:3]
                    for fmt in formats
                }
                raise ValueError(
                    f"{packed_qname}: packed MoE experts must be uniform "
                    f"within a layer (gridbook SPEC.md:288-291) but the "
                    f"allocation mixes {formats} across this stack's "
                    f"{len(present)} members — refusing to name it. "
                    f"Sample per format: {by_fmt}. Re-run the allocator with "
                    "the expert group constrained to one rung.")
            if formats[0] in SOURCE_PASSTHROUGH_EXPORT_FORMATS:
                # Keep the per-expert entries EXPANDED: they are the units this
                # route actually emits. `consumed` still claims the projections
                # so a narrower packed parent from the profile-less fallback
                # vocabulary cannot re-collapse them behind this decision.
                consumed.update(projections)
                continue
            for q in present:
                del collapsed[q]
            consumed.update(projections)
            collapsed[packed_qname] = formats[0]
            members_by_target[packed_qname] = member_qnames
            report["stacks"] = int(report["stacks"]) + 1
            report["members"] = int(report["members"]) + len(present)
    return collapsed, members_by_target, report


def _load_per_expert_config(path: str | Path) -> dict[str, str]:
    """Load the Tier-2 flat ``qname -> format`` allocation.

    The sibling counterfactual script writes the same entry spellings as a
    layer config but may include every dense/body row as well.  Only routed
    expert rows are consumed by this exporter mode; the ordinary layer config
    remains authoritative for all other tensors.
    """

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"--per-expert-config {path}: expected a qname->format JSON object"
        )
    return canonicalize_assignment(payload)


def _split_per_expert_assignment(
    assignment,
    per_expert_assignment,
    expert_groups,
    profile,
):
    """Build one packed target per ``(layer, family, format)``.

    Returns ``(assignment, members_by_target, plans, report)``.  ``plans`` is
    producer-internal metadata used to emit the proposed declaration after
    recipe names have been mapped into the physical checkpoint namespace.
    MXFP4_SOURCE members remain expanded because their checkpoint slices are
    copied verbatim; the plan, rather than ``source_passthrough.units``, owns
    their routing declaration.
    """

    merged = dict(assignment)
    members_by_target: dict[str, dict[tuple[str, int], str]] = {}
    plans: dict[str, dict[str, list[dict[str, object]]]] = {}
    report = {"stacks": 0, "members": 0, "format_groups": 0}
    matched_config_names: set[str] = set()
    packed_names = _packed_expert_param_names(profile)

    for prefix in sorted(expert_groups):
        group = expert_groups[prefix]
        consumed: set[str] = set()
        for packed_proj in sorted(
            packed_names,
            key=lambda name: (
                -len(_packed_expert_projection_names(profile, name)), name
            ),
        ):
            projections = _packed_expert_projection_names(profile, packed_proj)
            if any(projection in consumed for projection in projections):
                continue
            if any(projection not in group for projection in projections):
                continue
            try:
                all_members = _expert_member_qnames(
                    prefix, packed_proj, group, profile
                )
            except ValueError:
                continue
            configured = {
                qname for qname in all_members.values()
                if qname in per_expert_assignment
            }
            if not configured:
                continue
            missing = sorted(set(all_members.values()) - configured)
            if missing:
                raise ValueError(
                    f"{prefix} {_per_expert_family(profile, packed_proj)}: "
                    f"--per-expert-config covers {len(configured)} of "
                    f"{len(all_members)} members; missing e.g. {missing[:8]}"
                )
            matched_config_names.update(configured)
            consumed.update(projections)

            experts = sorted({expert_id for _projection, expert_id in all_members})
            formats_by_expert: dict[int, str] = {}
            for expert_id in experts:
                values = {
                    canonical_format_name(per_expert_assignment[
                        all_members[(projection, expert_id)]
                    ])
                    for projection in projections
                }
                if len(values) != 1:
                    detail = {
                        projection: per_expert_assignment[
                            all_members[(projection, expert_id)]
                        ]
                        for projection in projections
                    }
                    raise ValueError(
                        f"{prefix} {_per_expert_family(profile, packed_proj)} "
                        f"expert {expert_id}: coupled projections disagree: "
                        f"{detail}"
                    )
                formats_by_expert[expert_id] = values.pop()

            by_format: dict[str, list[int]] = {}
            for expert_id, format_name in formats_by_expert.items():
                _per_expert_format_wire_id(format_name)  # closed-menu gate
                by_format.setdefault(format_name, []).append(expert_id)

            packed_parent = f"{prefix}.{packed_proj}"
            if packed_parent in assignment:
                raise ValueError(
                    f"{packed_parent}: the layer config carries the packed "
                    "stack while --per-expert-config carries its expanded "
                    "members; describe the expert bank once"
                )
            for qname in all_members.values():
                merged.pop(qname, None)

            layer = _per_expert_layer_id(prefix)
            family = _per_expert_family(profile, packed_proj)
            family_plans = plans.setdefault(layer, {}).setdefault(family, [])
            mixed = len(by_format) > 1
            for format_name, expert_ids in sorted(by_format.items()):
                wire_id = _per_expert_format_wire_id(format_name)
                source_passthrough = format_name == "MXFP4_SOURCE"
                target = (
                    f"{packed_parent}.{_format_group_slug(wire_id)}"
                    if mixed and not source_passthrough
                    else packed_parent
                )
                subgroup_members = {
                    (projection, expert_id): all_members[(projection, expert_id)]
                    for projection in projections
                    for expert_id in expert_ids
                }
                if source_passthrough:
                    for member in subgroup_members.values():
                        merged[member] = format_name
                else:
                    if target in merged:
                        raise ValueError(
                            f"{target}: two per-expert format groups resolve "
                            "to the same packed tensor prefix"
                        )
                    merged[target] = format_name
                    members_by_target[target] = subgroup_members
                    report["stacks"] += 1
                    report["members"] += len(subgroup_members)
                family_plans.append({
                    "layer": layer,
                    "family": family,
                    "format": format_name,
                    "format_wire_id": wire_id,
                    "expert_ids": list(expert_ids),
                    "packed_parent": packed_parent,
                    "target": target if not source_passthrough else None,
                    "discriminated": bool(mixed and not source_passthrough),
                    "source_passthrough": source_passthrough,
                    "members": subgroup_members,
                })
                report["format_groups"] += 1

    unmatched = sorted(
        qname for qname in per_expert_assignment
        if ".experts." in qname and qname not in matched_config_names
    )
    if unmatched:
        raise ValueError(
            f"--per-expert-config contains {len(unmatched)} routed-expert "
            f"qname(s) that do not resolve in this checkpoint/profile, e.g. "
            f"{unmatched[:8]}"
        )

    # A declaration is needed only for a layer carrying more than one format
    # across its two families.  Single-format layers retain the targets,
    # codebook sharing, config and bytes of the legacy path byte-for-byte.
    plans = {
        layer: families
        for layer, families in plans.items()
        if len({
            str(entry["format"])
            for entries in families.values()
            for entry in entries
        }) > 1
    }
    return merged, members_by_target, plans, report


def _member_serialized_shapes(packed_qname, member_qnames, expert_groups,
                              skeleton, profile):
    """``{member recipe qname: decoded 2-D shape}`` for one collapsed stack.

    Read from the checkpoint rather than divided out of the stack shape, so a
    fused parent whose projections are NOT equal-width is described exactly."""
    if not member_qnames:
        return {}
    first_member = next(iter(member_qnames.values()))
    prefix = first_member.rsplit(".", 2)[0]
    group = expert_groups[prefix]
    out = {}
    for (proj, expert_id), member in member_qnames.items():
        base = group[proj][expert_id]
        out[member] = tuple(
            int(d) for d in skeleton.logical_shape(base + ".weight"))
    return out


def _assert_packed_plan_reconciles_to_recipe(recipe_formats, recipe_shapes,
                                             packed_payload,
                                             members_by_target, *, context,
                                             where):
    """The packed plan must equal the recipe's per-expert accounting EXACTLY,
    minus the deduplicated static-activation scalars.

    CB scales are per-expert/per-row/per-group, so a stack's packed bytes are
    the exact sum of its members' (the property that makes streaming a stack
    one expert at a time byte-identical in the first place). The one thing the
    collapse legitimately removes is ``input_global_scale``: the recipe carries
    one fp32 scalar per per-expert Linear, and gridbook's contract carries one
    per STACK. Anything else differing means the collapse changed the bytes,
    which it must never do."""
    recipe_payload = cb_assignment_payload_breakdown(
        recipe_formats, recipe_shapes, context=context)
    deduplicated = sum(
        len(members) - 1 for members in members_by_target.values())
    scalar_bytes = int(
        recipe_payload["input_global_scale_bytes"]
        - packed_payload["input_global_scale_bytes"]
    )
    expected_scalar_bytes = 0 if not deduplicated else scalar_bytes
    delta = int(recipe_payload["tensor_payload_bytes"]) - int(
        packed_payload["tensor_payload_bytes"])
    if delta != expected_scalar_bytes:
        raise AssertionError(
            f"{where}: packed expert stacks account for "
            f"{packed_payload['tensor_payload_bytes']}B against the recipe's "
            f"{recipe_payload['tensor_payload_bytes']}B — a difference of "
            f"{delta}B, but collapsing {len(members_by_target)} stack(s) may "
            f"only deduplicate {expected_scalar_bytes}B of "
            "input_global_scale scalars"
        )
    if int(packed_payload["index_bytes"]) != int(
        recipe_payload["index_bytes"]
    ) or int(packed_payload["fp4_scale_bytes"]) != int(
        recipe_payload["fp4_scale_bytes"]
    ) or int(packed_payload["fp8_row_scale_bytes"]) != int(
        recipe_payload["fp8_row_scale_bytes"]
    ):
        raise AssertionError(
            f"{where}: packed expert stacking moved CB weight bytes "
            f"(index/scale planes) relative to the recipe's per-expert "
            "accounting; per-expert and whole-stack packing must be "
            "byte-identical"
        )


def _packed_expert_col_weights(col_weights, members_by_target, profile):
    """Per-expert imatrix vectors -> the ``(E, 1, in)`` stack entry each packed
    target needs, returned as a NEW mapping (the per-expert entries survive for
    the render identity, which is keyed by them).

    A FUSED parent (``gate_up_proj`` = gate then up) has ONE input, so its two
    projections' vectors are two samples of the same per-column mean-square and
    are pooled by averaging. They are not identical in practice only because
    the probe caches each Linear's inputs separately under a row limit
    (DSv4-Flash layer 0: max |gate-up| ~ 0.3 against a vector norm ~ 4.6).
    Weighting one projection by the other's sample would be the actual error;
    per-row vectors cannot be expressed here at all, since the pack broadcasts
    ``(E, 1, in)`` across the whole stack."""
    out = dict(col_weights)
    for packed_qname, member_qnames in members_by_target.items():
        if packed_qname in out:
            continue
        projections = tuple(dict.fromkeys(
            projection for projection, _expert_id in member_qnames
        ))
        experts = sorted({e for _p, e in member_qnames})
        rows = []
        for e in experts:
            vecs = []
            for proj in projections:
                q = member_qnames[(proj, e)]
                if q not in col_weights:
                    raise ValueError(
                        f"{packed_qname}: CB stack member {q!r} has no "
                        "col_weights entry (no silent RTN)")
                vecs.append(torch.as_tensor(col_weights[q])
                            .reshape(-1).to(torch.float32))
            widths = {int(v.numel()) for v in vecs}
            if len(widths) != 1:
                raise ValueError(
                    f"{packed_qname}: expert {e} imatrix widths disagree "
                    f"across the fused projections {projections}: {widths}")
            rows.append(torch.stack(vecs).mean(dim=0) if len(vecs) > 1
                        else vecs[0])
        out[packed_qname] = torch.stack(rows).unsqueeze(1).contiguous()
    return out


def _stacked_source_weight(
        skeleton, profile, prefix, packed_proj, members, expert_ids=None) -> \
        torch.Tensor:
    """Materialise the full stacked source weight (E, out, in) for a packed
    expert group — used only where a stack must be resident (codebook
    training sampling); the packer streams per expert."""
    projections = _packed_expert_projection_names(profile, packed_proj)
    if expert_ids is None:
        expert_ids = range(_n_experts(members, projections))
    return torch.stack([
        _expert_weight(
            skeleton, profile, prefix, packed_proj, members, expert_id
        )
        for expert_id in expert_ids
    ])


def _n_experts(members: dict[str, dict[int, str]],
               projections: tuple[str, ...] | None = None) -> int:
    projections = tuple(projections or members)
    if not projections:
        raise ValueError("expert group has no projections")
    expected_ids = None
    for proj in projections:
        ids = members.get(proj)
        if ids is None:
            raise ValueError(f"expert group is missing projection {proj!r}")
        got = sorted(ids)
        if got != list(range(len(got))):
            raise ValueError(f"non-contiguous expert ids for {proj}: {got}")
        if expected_ids is None:
            expected_ids = got
        elif got != expected_ids:
            raise ValueError(
                f"expert ids differ across projections: {proj} has {got}, "
                f"expected {expected_ids}")
    return len(expected_ids)


def _expert_weight(skeleton, profile, prefix, packed_proj, members,
                   e, *, on_member=None) -> torch.Tensor:
    """Materialize one expert's packed projection using profile-declared
    projection names and order (for example LFM gate_up = ``w1`` then ``w3``).

    ``on_member(projection, expert_id, checkpoint_base, decoded)`` is called for
    each source tensor as it is decoded — the hook the render identity uses to
    verify a stack MEMBER BY MEMBER, since a checkpoint that stores experts
    per-expert has its source-value digests keyed per expert, not per stack."""
    projections = _packed_expert_projection_names(profile, packed_proj)
    tensors = []
    for p in projections:
        base = members[p][e]
        t = skeleton.dequant_weight(base + ".weight")
        if on_member is not None:
            on_member(p, e, base, t)
        tensors.append(t)
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)


# ---------------------------------------------------------------------------
# DELTA-EXPORT reuse: read a PRIOR artifact + decide byte-copy eligibility
# ---------------------------------------------------------------------------

class _PriorArtifact:
    """Read-only view of a PRIOR CB export for DELTA-EXPORT reuse.

    Exposes, for any tensor by name: presence, dtype, shape, and the exact
    ``(shard_path, file_offset, nbytes)`` byte slice (sharded via index.json or
    single-file). Also parses ``quant_config.json`` into the per-export-base CB
    ``(format, scheme)`` and loads the codebook sidecar — everything the
    eligibility gate needs to prove a re-encode would reproduce these bytes."""

    def __init__(self, prior_dir: str | Path):
        self.dir = Path(prior_dir)
        index = self.dir / "model.safetensors.index.json"
        self._single: Path | None = None
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            single = self.dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(
                    "reuse-prior: no model.safetensors[.index.json] under "
                    f"{self.dir}")
            self.weight_map = None
            self._single = single
        self._shard_hdr: dict[str, tuple[dict, int]] = {}
        qc_path = self.dir / "quant_config.json"
        if not qc_path.exists():
            raise FileNotFoundError(
                f"reuse-prior: no quant_config.json under {self.dir}")
        qc = json.loads(qc_path.read_text())
        # CB targets carry a "scheme"; stock/FP8_SOURCE groups do not.
        self.cb_by_base: dict[str, tuple[str, dict]] = {}
        self.stock_fmt_by_target: dict[str, str] = {}
        for g in qc.get("config_groups", {}).values():
            fmt = g.get("format")
            if "scheme" in g:
                for t in g.get("targets", []):
                    self.cb_by_base[t] = (fmt, g["scheme"])
            else:
                for t in g.get("targets", []):
                    self.stock_fmt_by_target[t] = fmt
        self.provenance = qc.get("provenance", {}) or {}
        self.scale_coding = qc.get("provenance", {}).get(
            "scale_coding") or "v1"
        self.codebooks: dict[str, torch.Tensor] = {}
        cbf = qc.get("codebook_file")
        if cbf and (self.dir / cbf).exists():
            from safetensors.torch import load_file as _lf
            self.codebooks = _lf(str(self.dir / cbf))

    def _shard_of(self, name: str) -> Path:
        if self._single is not None:
            return self._single
        return self.dir / self.weight_map[name]

    def _hdr(self, shard: Path) -> tuple[dict, int]:
        key = str(shard)
        if key not in self._shard_hdr:
            with open(shard, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(hlen))
            self._shard_hdr[key] = (hdr, 8 + hlen)
        return self._shard_hdr[key]

    def has(self, name: str) -> bool:
        if self.weight_map is not None:
            return name in self.weight_map
        hdr, _ = self._hdr(self._single)
        return name in hdr

    def _meta(self, name: str):
        shard = self._shard_of(name)
        hdr, data0 = self._hdr(shard)
        return hdr[name], data0, shard

    def dtype(self, name: str):
        meta, _, _ = self._meta(name)
        return _ST_DTYPE_INV.get(meta["dtype"])

    def shape(self, name: str) -> tuple[int, ...]:
        meta, _, _ = self._meta(name)
        return tuple(int(d) for d in meta["shape"])

    def raw_slice(self, name: str) -> tuple[Path, int, int]:
        """(shard_path, absolute file offset, nbytes) for a raw byte copy."""
        meta, data0, shard = self._meta(name)
        lo, hi = meta["data_offsets"]
        return shard, data0 + int(lo), int(hi) - int(lo)

    def read_bytes(self, name: str) -> bytes:
        shard, foff, nb = self.raw_slice(name)
        with open(shard, "rb") as f:
            f.seek(foff)
            return f.read(nb)

    def codebook_tensor(self, name: str) -> torch.Tensor | None:
        return self.codebooks.get(name)

    def matches_dtype_shape(self, name, dtype, shape) -> bool:
        return (self.has(name) and self.dtype(name) == dtype
                and self.shape(name) == tuple(int(d) for d in shape))


def _cb_reuse_reason(prior: _PriorArtifact, export_base: str, fmt: str,
                     cur_subset: dict, expected_outputs, group_cb_ok: bool):
    """Return None if this CB target is byte-copy eligible from ``prior``, else
    a short reason string. Eligible iff the prior assigns the SAME format +
    scheme signature, its codebook is byte-identical, and every planned output
    tensor already exists in the prior at EXACTLY the planned dtype+shape."""
    entry = prior.cb_by_base.get(export_base)
    if entry is None:
        return "not_in_prior"
    pfmt, pscheme = entry
    if pfmt != fmt:
        return "format_changed"
    pscheme_norm = cb_scheme_reuse_signature(pscheme)
    if pscheme_norm != cur_subset:
        return "scheme_changed"
    if not group_cb_ok:
        return "codebook_mismatch"
    for name, dtype, shape in expected_outputs:
        if not prior.has(name):
            return "tensor_missing"
        if not prior.matches_dtype_shape(name, dtype, shape):
            return "dtype_shape_mismatch"
    return None


def _current_imatrix_sha(col_weights: dict[str, torch.Tensor]) -> str:
    """The imatrix hash exactly as the shared config builder computes it —
    used to diagnose whether the reuse prior shares this calibration."""
    ih = hashlib.sha256()
    for q in sorted(col_weights):
        ih.update(q.encode())
        ih.update(col_weights[q].to(torch.float32).cpu().numpy().tobytes())
    return ih.hexdigest()


def _reuse_verify_and_report(prior, reuse, reuse_verify, reuse_prior,
                             col_weights, scale_coding, counts):
    """MANDATORY reuse safety gate (runs BEFORE any bytes are written): fresh
    re-encode ``reuse_verify`` random copy-eligible CB targets and byte-compare
    against what would be copied from the prior; ANY mismatch means the
    determinism contract broke and aborts the export. Also logs the copied/
    encoded/ineligible summary and folds ``reuse_*`` counters into ``counts``."""
    import random

    cur_sha = _current_imatrix_sha(col_weights)
    prior_sha = prior.provenance.get("imatrix_sha256")
    imatrix_match = (prior_sha is not None and prior_sha == cur_sha)
    if prior_sha is not None and not imatrix_match:
        print("[export-cb-stream] WARNING reuse-prior imatrix_sha256 differs "
              f"(prior {prior_sha[:12]} vs current {cur_sha[:12]}) — encoding "
              "inputs may have changed; copied bytes rest on the verification "
              "sample below. Double-check --reuse-prior points at the SAME "
              "source+calibration.", flush=True)

    pool = reuse["verify_pool"]
    n = min(int(reuse_verify), len(pool))
    if n > 0:
        # Deterministic sample (reproducibility gate): seed from the stable set
        # of eligible bases so a resumed run verifies the same targets.
        key = "|".join(sorted(c["base"] for c in pool))
        rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16],
                                16))
        for cand in rng.sample(pool, n):
            fresh = cand["fresh"]()
            for name, dtype, shape in cand["specs"]:
                fb = _raw_bytes(fresh[name])
                pb = prior.read_bytes(name)
                if fb != pb:
                    raise RuntimeError(
                        "[export-cb-stream] REUSE VERIFICATION FAILED: fresh "
                        f"re-encode of {name} does NOT byte-match the prior "
                        f"artifact ({len(fb)}B vs {len(pb)}B copied). The "
                        "determinism/RESUME contract is broken for this "
                        "(source, imatrix, codebook, scheme) — refusing to "
                        "ship reused bytes. Re-run WITHOUT --reuse-prior, or "
                        "point it at the artifact this allocation derives from.")
            reuse["verified"] += 1
        print(f"[export-cb-stream] reuse verify OK: {reuse['verified']} "
              f"sampled copy target(s) byte-match the prior", flush=True)

    print(f"[export-cb-stream] reuse-prior {reuse_prior}: "
          f"copied {reuse['copied']} / encoded {reuse['encoded']} targets; "
          f"imatrix {'MATCH' if imatrix_match else 'differ/absent'}; "
          f"scale_coding prior={prior.scale_coding} current={scale_coding}",
          flush=True)
    if reuse["reasons"]:
        print("[export-cb-stream] reuse re-encode reasons: "
              f"{dict(sorted(reuse['reasons'].items()))}", flush=True)

    counts["reuse_copied"] = reuse["copied"]
    counts["reuse_encoded"] = reuse["encoded"]
    counts["reuse_verified"] = reuse["verified"]
    for reason, c in reuse["reasons"].items():
        counts[f"reuse_ineligible_{reason}"] = c


# ---------------------------------------------------------------------------
# Streaming export
# ---------------------------------------------------------------------------

def assert_routes_reconcile(*, cb_units, passthrough_units, cb_tensors,
                            passthrough_tensors, cb_modules,
                            passthrough_modules, attested) -> None:
    """Producer-side invariant; none of this is written to the artifact.

    This is what ``cb_activation_contract`` used to buy on the wire, kept as an
    internal check now that the field is gone:

      * NO unit is claimed by both gridbook's codec and a passthrough. That is
        the load-time refusal the consumer names, and the exporter is the only
        place that can see it across BOTH namespaces — the declaration speaks
        recipe names and the config groups speak serialized ones, so the
        consumer's own same-string test cannot catch it on DSv4.
      * No TENSOR is emitted by both routes either. The unit check is the
        contract-level statement; this is the physical one that would catch a
        namespace bug the unit ids happened to hide.
      * The K0.2 attestation covers exactly the CB routed groups: a delegated
        group must never be attested (it has no CB activation contract at all),
        and an attested module must be a CB group. THAT is what makes a
        delegated group's ABSENCE from the K0.2 record a declaration rather
        than a dropped attestation.

    A routed-expert group left entirely on BF16 passthrough is on neither route
    by definition and is correctly absent from both module sets.
    """
    contested = sorted(set(cb_units) & set(passthrough_units))
    if contested:
        raise AssertionError(
            f"{contested[:5]} are claimed by BOTH a CB config group and the "
            "source_passthrough declaration; a unit is decoded by gridbook's "
            "codec or handed to the model's own loader, never both")
    shared = sorted(set(cb_tensors) & set(passthrough_tensors))
    if shared:
        raise AssertionError(
            f"{shared[:5]} are emitted by both a CB target and a "
            "source-passthrough target")
    both_routed = sorted(set(attested) & set(passthrough_modules))
    if both_routed:
        raise AssertionError(
            f"{both_routed[:5]} are attested by the routed-MoE activation "
            "contract AND declared source-passthrough; a delegated group has "
            "no CB activation contract to attest")
    unattributed = sorted(set(attested) - set(cb_modules))
    if unattributed:
        raise AssertionError(
            f"the routed-MoE activation contract attests {unattributed[:5]}, "
            "which no CB expert unit claims")


#: Block geometry the ``fp8_e4m3_ue8m0_block128`` wire id NAMES. A checkpoint
#: whose declared block is anything else stores a different contract under the
#: same element dtype, so it must not borrow this id.
_FP8_UE8M0_DECLARED_BLOCK = (128, 128)

#: Registry format a floor block-FP8 unit is declared as. One spelling, so the
#: wire id it maps to stays the table's decision rather than this module's.
_FP8_BLOCK_UE8M0_FORMAT = "FP8_BLOCK_UE8M0_SOURCE"

#: Env spelling of ``--allow-route-pending-passthrough``, so a driver script
#: can pass the acknowledgement without editing the command line it builds.
_ROUTE_PENDING_ACK_ENV = "PQ_ALLOW_ROUTE_PENDING"

#: Env spelling of ``--exclude-namespace``: comma-separated tensor-name
#: prefixes to OMIT from the artifact entirely.
_EXCLUDE_NAMESPACES_ENV = "PQ_EXPORT_EXCLUDE_NAMESPACES"


def _exclude_namespaces_from_env() -> tuple[str, ...]:
    """Read the env form of the namespace exclusion list. Empty by default.

    Empty means "exclude nothing", which is the pre-existing behaviour, so an
    unset or blank variable cannot change what an export produces.
    """

    raw = os.environ.get(_EXCLUDE_NAMESPACES_ENV, "")
    prefixes = tuple(part.strip() for part in raw.split(",") if part.strip())
    if prefixes:
        print(f"[export-stream] {_EXCLUDE_NAMESPACES_ENV} -> omitting "
              f"namespace(s) {list(prefixes)} from this artifact entirely.",
              file=sys.stderr, flush=True)
    return prefixes


def _validate_namespace_exclusions(
    exclude_namespaces,
    *,
    assignment,
    profile,
    budget_stamp=None,
) -> tuple[str, ...]:
    """Normalize the exclusion list, refusing anything the recipe allocated.

    OMISSION IS ONLY LEGAL FOR THE FLOOR. A verbatim tensor is one the
    allocator never reasoned about, so dropping it changes the artifact's
    contents and nothing else. An ALLOCATED unit is different in kind: the DP
    priced it, spent budget on it, and the selection's achieved-bits and
    predicted loss are both computed as though it ships. Silently omitting one
    would make the artifact disagree with the recipe that justifies it, and
    the discrepancy would surface as a missing-weights error at load, long
    after the number it invalidated was reported. So this is a hard refusal
    rather than a warning: an exclusion that collides with the recipe means
    the operator meant something else.

    Both namespaces are checked, because a prefix is written in whichever
    spelling the operator has in hand (``mtp.`` is a checkpoint spelling; the
    recipe would say ``model.mtp.``), and a prefix that misses only because it
    was written in the other vintage would be a silent no-op.

    When the recipe carries a whole-artifact budget stamp, the exclusion set
    must additionally equal the set the price was computed without. Those are
    two halves of one statement -- the allocator hands the excluded bytes to
    the body only if the exporter really omits them -- and until they were
    checked against each other, making just one of them was silent in the
    direction that costs quality.
    """

    prefixes = tuple(
        str(prefix).strip() for prefix in (exclude_namespaces or ())
        if str(prefix).strip()
    )
    # Before the empty-list shortcut, because the dangerous direction is the
    # one with NO exclusions here: a price computed without a namespace, spent
    # on the body, and then an export that writes the namespace anyway.
    assert_exclusions_match_budget_stamp(
        budget_stamp, prefixes,
        where="export_nvfp4_cb_streaming namespace exclusions")
    if not prefixes:
        return ()

    collisions: dict[str, list[str]] = {}
    for qname in assignment:
        spellings = {str(qname)}
        checkpoint = _canonical_qname(str(qname), profile)
        if checkpoint:
            spellings.add(checkpoint)
        source = getattr(profile, "source_tensor_name", None)
        if callable(source):
            try:
                spellings.add(source(str(qname)))
            except Exception:              # pragma: no cover - defensive
                pass
        for prefix in prefixes:
            if any(name.startswith(prefix) for name in spellings):
                collisions.setdefault(prefix, []).append(str(qname))

    if collisions:
        detail = "; ".join(
            f"{prefix!r} matches {len(names)} allocated unit(s) "
            f"e.g. {sorted(names)[:3]}"
            for prefix, names in sorted(collisions.items())
        )
        raise ValueError(
            f"refusing to exclude a namespace the recipe allocates: {detail}. "
            f"Namespace exclusion omits tensors from the artifact entirely, "
            f"which is only sound for FLOOR units the allocator never priced. "
            f"An allocated unit's bytes are already counted in the "
            f"selection's achieved bits and predicted loss, so dropping it "
            f"here would make the artifact contradict the recipe that "
            f"justifies it. Re-run the allocation without these units, or "
            f"narrow the exclusion prefix.")
    return prefixes


def _route_pending_ack_from_env() -> bool:
    """Read the env form of the route-pending acknowledgement.

    DEFAULTS OFF, and off for every value except exactly ``"1"`` — an
    acknowledgement that "0"/"false"/"" could switch on would be the opposite
    of deliberate. It announces itself on stderr when it fires, because the
    one failure mode an env knob has that a flag does not is being inherited
    silently by an invocation nobody meant to acknowledge for.
    """

    if os.environ.get(_ROUTE_PENDING_ACK_ENV) != "1":
        return False
    print(f"[export-stream] {_ROUTE_PENDING_ACK_ENV}=1 -> "
          f"--allow-route-pending-passthrough is ON for this invocation; "
          f"route-pending passthrough units will ship with the "
          f"acknowledgement recorded in the artifact provenance.",
          file=sys.stderr, flush=True)
    return True


def _floor_block_fp8_units(
    skeleton,
    *,
    emitted_bases: set[str],
    consumed_expert_bases: set[str],
    claimed_qnames: set[str],
    profile,
    subset_prefixes,
    excluded_namespaces: tuple[str, ...] = (),
) -> tuple[dict[str, str], dict[str, str]]:
    """Block-FP8 units the recipe never allocated, which must be DECLARED.

    Returns ``({checkpoint qname: scale checkpoint key},
    {scale checkpoint key: checkpoint qname})``.

    THE BUG THIS EXISTS TO CLOSE. A weight that no allocation target claims
    falls to the verbatim copy loop below. That loop was written for BF16
    norms and buffers, for which "copy the tensor and list it in ``ignore``"
    is exactly right. For a block-FP8 weight it was silently wrong in two
    compounding ways: the loop skips ``.scale`` siblings as "consumed with
    their fp8 weight" — but nothing consumed them, because the weight was
    never a target — so the scale plane was DROPPED; and the weight was then
    declared ``ignore``, i.e. unquantized. A consumer honouring that
    declaration allocates a bf16 parameter, the size assertion passes because
    the element counts match, and the fp8 bytes are cast to bf16 with no scale
    applied. Every block is then wrong by its own power of two, with no error
    raised anywhere. Measured on DSv4-Flash: 43 ``attn.wo_a`` + 21
    ``attn.indexer.wq_b`` units, 1.44 GB, silently corrupted.

    So a floor block-FP8 unit ships its weight AND its scale, and is declared
    ``fp8_e4m3_ue8m0_block128`` rather than ignored — the same delegated-native
    route an ALLOCATED passthrough unit of that format would get. Nothing about
    the bytes differs between the two cases; only whether the DP happened to
    choose the unit, which is not a property the consumer can see or should
    have to.

    MEMBERSHIP IS NARROW, AND DELIBERATELY SO. A unit qualifies only when the
    checkpoint pairs an ``F8_E4M3`` weight with an ``F8_E8M0`` scale over the
    declared 128x128 block. That excludes, by construction rather than by
    name: MXFP4 nibble-packs (int8/uint8 elements), BF16/F32 verbatim tensors
    (no scale plane, and for which the ``ignore`` cast IS correct), and any
    architecture whose declared block is not 128x128. Prefixes the profile
    ships verbatim-and-undeclared (``source_passthrough_prefixes``, e.g. DSv4's
    ``mtp.*``) are excluded too: those are units no serving stack builds, so
    declaring them would assert a route nobody exercises.
    """

    scale_map = skeleton._fp8_scale_inv_map
    block = getattr(scale_map, "block", None)
    _verbatim = getattr(profile, "source_passthrough_prefixes", None)
    verbatim_prefixes = tuple(_verbatim()) if callable(_verbatim) else ()

    units: dict[str, str] = {}
    # Scan the CHECKPOINT, not `scale_map`. The map is keyed by LIVE MODEL
    # NAME, and a profile may legitimately map a shipped checkpoint tensor to
    # no model name at all: DSv4 returns None for `attn.compressor.*` and
    # `attn.indexer.*` because probe mode disables those modules
    # (model_profiles/deepseek_v4.py). Those tensors still ship, so a
    # probe-time graph decision must not decide how their bytes are DECLARED.
    #
    # Iterating the map silently skipped every indexer unit -- measured
    # 2026-08-08 on DSv4-Flash: `wo_a` 43 entries in the map, `indexer` 0 of
    # 33368 -- so 21 `attn.indexer.wq_b` weights fell to the verbatim loop and
    # were declared `ignore`, which is exactly the silent corruption this
    # function exists to prevent. That is why the docstring above already
    # listed indexer.wq_b as covered when it was not: the intent was right and
    # the iteration source was wrong. Scanning the checkpoint makes the
    # docstring's stated contract ("the checkpoint pairs an F8_E4M3 weight
    # with an F8_E8M0 scale") literally what the code does.
    for scale_key in sorted(skeleton.keys()):
        if not scale_key.endswith(".scale"):
            # Legacy `.weight_scale_inv` checkpoints normalize through the
            # FP8_SOURCE lane and never reach this fallback.
            continue
        ckpt_qname = scale_key[: -len(".scale")]
        weight_key = ckpt_qname + ".weight"
        if weight_key not in skeleton:
            continue
        if subset_prefixes is not None and not any(
                weight_key.startswith(p) for p in subset_prefixes):
            continue
        if any(ckpt_qname.startswith(p) for p in excluded_namespaces):
            continue                      # omitted from the artifact entirely
        if any(ckpt_qname.startswith(p) for p in verbatim_prefixes):
            continue                      # profile ships these undeclared
        if weight_key in emitted_bases or weight_key in consumed_expert_bases:
            continue                      # a target already claimed it
        canon = _canonical_qname(ckpt_qname, profile)
        if ckpt_qname in claimed_qnames or canon in claimed_qnames:
            continue
        if skeleton.get_dtype(weight_key) is not torch.float8_e4m3fn:
            continue                      # not a block-FP8 element plane
        if skeleton.get_dtype(scale_key) is not torch.float8_e8m0fnu:
            continue                      # not a UE8M0 scale plane
        # Geometry is asserted rather than assumed: the wire id NAMES a
        # 128x128 block, and a declaration that misdescribes the grid is
        # worse than no declaration at all — it loads, and it is wrong.
        if tuple(block or ()) != _FP8_UE8M0_DECLARED_BLOCK:
            raise ValueError(
                f"{ckpt_qname}: unallocated block-FP8 unit, but the "
                f"checkpoint declares weight_block_size {block!r} rather "
                f"than {list(_FP8_UE8M0_DECLARED_BLOCK)}. The "
                f"fp8_e4m3_ue8m0_block128 wire id names the 128x128 grid, so "
                f"this unit cannot be declared under it. Shipping it "
                f"undeclared is refused: it would be cast to bf16 without "
                f"its scale.")
        w_shape = skeleton.get_shape(weight_key)
        s_shape = skeleton.get_shape(scale_key)
        expected = tuple(-(-int(d) // b)
                         for d, b in zip(w_shape, _FP8_UE8M0_DECLARED_BLOCK))
        if tuple(s_shape) != expected:
            raise ValueError(
                f"{ckpt_qname}: block-FP8 scale grid {tuple(s_shape)} does "
                f"not match weight {tuple(w_shape)} at "
                f"{_FP8_UE8M0_DECLARED_BLOCK} (expected {expected}). A "
                f"transposed grid is numel-compatible and would mis-scale "
                f"every block.")
        units[ckpt_qname] = scale_key
    return units, {scale: unit for unit, scale in units.items()}


def _reject_disabled_reuse_before_output_transaction(function):
    """Preserve the quarantine gate ahead of any resume/output interpretation."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get("reuse_prior") is not None:
            raise RuntimeError(
                "DELTA-EXPORT reuse is disabled: the prior artifact is not "
                "bound to exact source-content, imatrix, codebook, and "
                "exporter-ABI identity for every copied tensor. Re-encode "
                "into a fresh output directory."
            )
        return function(*args, **kwargs)

    return wrapped


@_reject_disabled_reuse_before_output_transaction
@transactional_directory_output(
    source_parameter="model_dir",
    output_parameter="out_dir",
    where="export_nvfp4_cb_streaming",
)
def export_nvfp4_cb_streaming(
    model_dir: str | Path,
    layer_config_path: str | Path,
    out_dir: str | Path,
    col_weights: dict[str, torch.Tensor],
    *,
    shared_codebook_spec: dict | None = None,
    device: str | None = None,
    scale_sweep: bool = True,
    scale_coding: str = cb.SCALE_CODING_TWO_TIER,
    subset_prefixes: list[str] | None = None,
    reuse_prior: str | Path | None = None,
    reuse_verify: int = 3,
    allow_unstamped_research: bool = False,
    allow_research_cost_selection: bool = False,
    allow_route_pending_passthrough: bool = False,
    allow_per_role_books: bool = False,
    allow_unbacked_route: str | None = None,
    non_native_target: str | None = None,
    exclude_namespaces: list[str] | tuple[str, ...] | None = None,
    activation_cache_dir: str | Path | None = None,
    activation_scale_policy: str | None = None,
    per_expert_config_path: str | Path | None = None,
    warm_state_dir: str | Path | None = None,
    warm_verify_sample: int = 32,
    dspark_cb_sidecar: bool = False,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
) -> dict[str, int]:
    """Streaming counterpart of :func:`export_nvfp4_cb.export_nvfp4_cb`. Same
    signature + container; peak residency ~= one source tensor + codebooks.
    See the module docstring for the scope of this milestone.

    ``subset_prefixes`` (opt-in) scopes the export to a subset of the model:
    the passthrough copies ONLY checkpoint tensors whose name starts with one of
    the prefixes, and every allocation target must resolve to an export base
    within them (else the allocation and the declared subset disagree — fail
    fast). Default ``None`` = whole-model passthrough, byte-identical to before.
    Used to export just the MTP sidecar (``model.layers.80.``) without dragging
    the ~550 GB body through as bf16 passthrough.

    ``allow_route_pending_passthrough`` overrides the ship gate on a
    source-passthrough rung whose serve route has not been validated
    (``allocator_candidates.ROUTE_PENDING_PASSTHROUGH_FORMATS``). The rung stays
    on the allocator's menu on purpose — an allocation that wants it is
    reporting a serving gap worth seeing — so the refusal lives here, at the
    ship step, and the override is recorded in the artifact's provenance rather
    than only in the operator's shell history.

    ``allow_per_role_books`` overrides the split-book ship gate (campaign rule
    R1): a fused routed weight whose scheme would name more than one codebook
    refuses unless this is passed, and passing it stamps the fact onto the
    shipcard. The keying comes from the bundle's own record of how each routed
    book was burned, never from a flag here.
    ``allow_unbacked_route`` and ``non_native_target`` are the two dispositions
    principle 9 allows for a selected unit with no backed serving route under
    the pinned Gridbook release (campaign rule R3). Both are STRINGS, not
    booleans, and both are stamped: the first is the REASON this artifact ships
    over an unbacked route, the second names the target platform whose native
    lane the artifact is not claiming. A bare ``1`` documents nothing, and this
    record is what a reviewer reads when the artifact turns out to serve badly.
    They are a separate gate from ``allow_route_pending_passthrough`` above,
    which governs source-passthrough rungs rather than lane routes.

    ``reuse_prior`` is reserved but currently fails closed. The prior gate did
    not bind exact source content, treated an imatrix mismatch as a warning,
    sampled only some CB targets, and copied stock targets on dtype/shape alone.
    Reuse may return only after one immutable producer-input identity covers
    source bytes, imatrix, codebooks, scheme, and exporter ABI for every copied
    tensor. ``reuse_verify`` is retained only for CLI compatibility while reuse
    is blocked.

    ``per_expert_config_path`` enables the proposed split-stack producer ABI.
    It is a flat qname-to-format mapping; routed expert rows override the base
    layer config while all ordinary rows continue to come from that config.

    ``dspark_cb_sidecar`` is the explicit DeepSeek-V4 DSpark draft producer.
    It accepts the closed three-stage ``mtp.*`` source payload, keeps emitted
    tensors in that physical checkpoint namespace, and writes Gridbook config
    targets in vLLM's construction namespace
    (``model.layers.{num_hidden_layers + stage}.*``).  The ordinary source-
    passthrough overlay and this quantized sidecar mode are mutually exclusive.
    A separate ``/draft`` artifact is emitted; the target artifact is never
    rewritten or linked into the draft.

    ``shard_bytes`` is the per-container byte budget, 1 GiB by default, the
    same value and the same partition rule the compressed-tensors lane ships
    (``run-pipeline.sh``'s ``EXPORT_SHARD_BYTES``).  One resulting shard keeps
    the legacy ``model.safetensors``; more than one publishes
    ``model-XXXXX-of-YYYYY.safetensors`` plus ``model.safetensors.index.json``,
    which is the layout a stock HF/vLLM loader already reads.  There is no zero
    sentinel: pass a budget at least as large as the artifact to reproduce a
    pre-2026-08-21 single-container CB export.
    """
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    source_model_identity = _source_model_identity_from_env(model_dir)
    _require_production_source_model_identity(
        model_dir,
        source_model_identity,
        allow_unstamped_research=allow_unstamped_research,
    )
    if reuse_prior is not None:
        raise RuntimeError(
            "DELTA-EXPORT reuse is disabled: the prior artifact is not bound "
            "to exact source-content, imatrix, codebook, and exporter-ABI "
            "identity for every copied tensor. Re-encode into a fresh output "
            "directory."
        )
    if int(warm_verify_sample) < 0:
        raise ValueError("warm_verify_sample must be >= 0")
    if scale_coding not in (cb.SCALE_CODING_V1, cb.SCALE_CODING_TWO_TIER):
        raise ValueError(f"unknown scale_coding {scale_coding!r}")
    out_dir = prepare_fresh_export_directory(
        model_dir,
        out_dir,
        where="export_nvfp4_cb_streaming",
    )
    subset_prefixes = list(subset_prefixes) if subset_prefixes else None
    prior = _PriorArtifact(reuse_prior) if reuse_prior else None
    reuse = {"copied": 0, "encoded": 0, "verified": 0,
             "reasons": Counter(), "verify_pool": []}
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = shared_codebook_spec or {}
    source = str(spec.get("source", "lattice")).lower()
    if source not in ("lattice", "learned"):
        raise ValueError(f"shared_codebook_spec source must be lattice/learned")
    _env_cb_context = cb_serialization_context_from_env()
    _scoped_bundle_export = (
        effective_codebook_source_scope(_env_cb_context) != "none"
    )
    if _scoped_bundle_export:
        if source != _env_cb_context.codebook_source:
            raise ValueError(
                "export_nvfp4_cb_streaming: shared_codebook_spec source "
                "differs from CB_CODEBOOK_SOURCE_SCOPE/CB_CODEBOOK_BUNDLE"
            )
        source = _env_cb_context.codebook_source
    elif source == "learned" and bool(spec.get("train", False)) and not (
        allow_unstamped_research
    ):
        raise ValueError(
            "export_nvfp4_cb_streaming: production export-time learned-"
            "codebook retraining is forbidden; build CB_CODEBOOK_BUNDLE "
            "before cost. The legacy trainer is available only with "
            "allow_unstamped_research=True."
        )

    assignment = load_assignment(layer_config_path)
    _recipe_assignment = dict(assignment)
    _recipe_payload = json.loads(Path(layer_config_path).read_text())
    _recipe_cb_context_stamp, _recipe_cb_tensor_stamps = (
        cb_serialization_metadata_from_assignment_payload(_recipe_payload)
    )
    _recipe_meta = _recipe_payload.get("__prismaquant__", {})
    from prismaquant.research_cost_acceptance import (
        enforce_research_export_acknowledgement,
    )
    _research_cost_selection = enforce_research_export_acknowledgement(
        _recipe_payload,
        acknowledged=allow_research_cost_selection,
        where="export_nvfp4_cb_streaming",
    )
    _recipe_cb_render_identity = _recipe_payload.get("cb_render_identity")
    if _recipe_cb_render_identity is None and isinstance(_recipe_meta, dict):
        _recipe_cb_render_identity = _recipe_meta.get("cb_render_identity")
    _dspark_render_recipe = (
        _recipe_meta.get("dspark_render_recipe")
        if isinstance(_recipe_meta, dict)
        else None
    )
    production_recipe_stamped = (
        _recipe_cb_context_stamp is not None or bool(_recipe_cb_tensor_stamps)
    )
    _claimed_activation_contract = (
        _recipe_cb_context_stamp.get("activation_contract")
        if isinstance(_recipe_cb_context_stamp, dict)
        else None
    )
    if _claimed_activation_contract not in (
        None,
        NVFP4_ACTIVATION_CONTRACT_SCHEMA,
    ):
        raise ValueError(
            "export_nvfp4_cb_streaming: unsupported activation contract "
            f"{_claimed_activation_contract!r}"
        )
    _whole_artifact_budget = whole_artifact_budget_from_assignment_payload(
        _recipe_payload,
        where="export_nvfp4_cb_streaming layer config",
        assignment=assignment,
    )
    skeleton = _LazySkeleton(model_dir)
    try:
        profile = detect_profile(str(model_dir))
    except Exception:
        profile = None
    source_config_path = model_dir / "config.json"
    source_config = (
        json.loads(source_config_path.read_text())
        if source_config_path.exists() else {}
    )
    discovered_dspark_source_overlay = discover_dspark_source_overlay(
        skeleton, source_config
    )
    dspark_hybrid_source_mapping: dict[str, str] = {}
    if dspark_cb_sidecar:
        if discovered_dspark_source_overlay is None:
            raise ValueError(
                "--dspark-cb-sidecar requires the validated released "
                "DeepSeek-V4 three-stage mtp.* source payload"
            )
        if subset_prefixes != ["mtp."]:
            raise ValueError(
                "--dspark-cb-sidecar requires exactly --subset-prefix mtp.; "
                "a draft sidecar must contain all and only the atomic mtp.* "
                "checkpoint namespace"
            )
        dspark_hybrid_source_mapping = dspark_cb_source_passthrough_mapping(
            source_config
        )
        for physical_source in dspark_hybrid_source_mapping:
            source_format = (
                discovered_dspark_source_overlay.physical_targets.get(
                    physical_source
                )
            )
            if source_format != _FP8_BLOCK_UE8M0_FORMAT:
                raise ValueError(
                    f"validated DSpark source has no "
                    f"{_FP8_BLOCK_UE8M0_FORMAT} contract for "
                    f"{physical_source}"
                )
            prior_format = assignment.get(physical_source)
            if physical_source == "mtp.0.main_proj":
                if prior_format not in (None, source_format):
                    raise ValueError(
                        f"{physical_source}: DSpark glue must remain "
                        f"{source_format}, got {prior_format!r}"
                    )
                # main_proj is not allocator-owned, so make its immutable
                # source route explicit here.
                assignment[physical_source] = source_format
            elif prior_format != source_format:
                raise ValueError(
                    f"{physical_source}: grouped-BMM wo_a must be explicitly "
                    f"assigned {source_format}; CB Linear has no grouped-BMM "
                    f"semantics, got {prior_format!r}"
                )

    # Preserve the allocator's expanded, per-Linear namespace for independent
    # release-route replay.  The serializer below collapses routed experts to
    # physical stacks, which is correct for bytes but cannot replace the
    # certified serving-unit member ledger.
    finalized_tensor_formats = dict(assignment)
    dspark_source_overlay = discovered_dspark_source_overlay
    expert_groups = (
        _plan_dspark_cb_expert_stacks(skeleton, source_config)
        if dspark_cb_sidecar
        else _plan_expert_stacks(skeleton, profile)
    )
    # The planner's contract is that "whichever spelling matched becomes the
    # group key, so `_resolve_target` finds the group under the recipe name"
    # (see `_plan_expert_stacks`). For a WRAPPED source (Qwen3.5-VLM) the
    # skeleton speaks the live module tree (`language_model.model.*`), which
    # is neither the recipe (`model.*`) nor the checkpoint
    # (`model.language_model.*`) spelling, so the matched key is in a
    # namespace nothing downstream can bridge: the coverage gate KeyErrors on
    # uniform groups and every consumed per-expert source ships verbatim into
    # `ignore`. Normalize ONLY such keys to the RECIPE spelling, derived from
    # the group's own member tensors (real checkpoint names, which
    # `_canonical_qname` always maps); groups already keyed by a recipe or
    # checkpoint spelling — DSv4-class sources — are left exactly as planned.
    # The recipe->checkpoint prefix bridge is recorded for emission either
    # way: packed-stack tensor NAMES must carry the checkpoint prefix (the
    # packed parent is not a checkpoint leaf, so it cannot be mapped
    # directly), and gridbook's top-level loader bridges checkpoint
    # spellings only.
    _canon_to_ckpt_prefix: dict[str, str] = {}
    if not dspark_cb_sidecar:
        def _member_prefixes(_projs):
            for _members in _projs.values():
                for _member in _members.values():
                    _ck = str(_member).rsplit(".", 2)[0]
                    _cm = _canonical_qname(str(_member), profile)
                    _rc = _cm.rsplit(".", 2)[0] if _cm else None
                    return _rc, _ck
            return None, None

        _rekeyed = {}
        for _prefix, _projs in expert_groups.items():
            _recipe, _ck = _member_prefixes(_projs)
            if _recipe is not None and _ck is not None:
                _canon_to_ckpt_prefix.setdefault(_recipe, _ck)
            _key = _prefix
            if (_recipe is not None
                    and _prefix not in (_recipe, _ck)
                    and _recipe not in expert_groups
                    and _recipe not in _rekeyed):
                _key = _recipe
            _rekeyed[_key] = _projs
        expert_groups = _rekeyed
    # The allocator writes its layer_config EXPANDED per tensor even though it
    # decided each expert group atomically, so a per-expert checkpoint arrives
    # as one entry per (expert, projection). Gridbook only names stacks. Do the
    # reduction here, once, before anything reads the assignment.
    per_expert_plans: dict[str, dict[str, list[dict[str, object]]]] = {}
    if per_expert_config_path is not None:
        per_expert_assignment = _load_per_expert_config(per_expert_config_path)
        finalized_tensor_formats.update({
            qname: fmt
            for qname, fmt in per_expert_assignment.items()
            if ".experts." in qname
        })
        (
            assignment,
            expert_stack_members,
            per_expert_plans,
            expert_stack_report,
        ) = _split_per_expert_assignment(
            assignment,
            per_expert_assignment,
            expert_groups,
            profile,
        )
    else:
        assignment, expert_stack_members, expert_stack_report = (
            _collapse_per_expert_assignment(
                assignment, expert_groups, profile
            )
        )
    # Namespace exclusion is validated against the COLLAPSED assignment, i.e.
    # the units as the allocator actually decided them, so a per-expert entry
    # cannot hide a collision behind its expanded spelling.
    excluded_namespaces = _validate_namespace_exclusions(
        exclude_namespaces if exclude_namespaces is not None
        else _exclude_namespaces_from_env(),
        budget_stamp=_whole_artifact_budget,
        assignment=assignment,
        profile=profile,
    )
    if dspark_cb_sidecar and excluded_namespaces:
        raise ValueError(
            "--dspark-cb-sidecar does not permit namespace exclusions; the "
            "three-stage draft payload is atomic"
        )
    if dspark_source_overlay is not None:
        mtp_names = {
            name for name in skeleton.keys() if str(name).startswith("mtp.")
        }
        included_mtp_names = (
            mtp_names if subset_prefixes is None else {
                name for name in mtp_names
                if any(name.startswith(prefix) for prefix in subset_prefixes)
            }
        )
        if not included_mtp_names:
            # A body-only subset contains no physical draft tensors, so it
            # must not promise a draft construction overlay.
            dspark_source_overlay = None
        elif included_mtp_names != mtp_names:
            raise ValueError(
                "refusing a partial DSpark subset: the source overlay is an "
                f"atomic three-stage contract, but {len(included_mtp_names)} "
                f"of {len(mtp_names)} mtp.* tensors would be emitted"
            )
    if dspark_source_overlay is not None:
        mtp_names = {
            name for name in skeleton.keys() if str(name).startswith("mtp.")
        }
        excluded_mtp_names = {
            name for name in mtp_names
            if any(name.startswith(prefix) for prefix in excluded_namespaces)
        }
        if excluded_mtp_names == mtp_names:
            # A deliberate whole-namespace body-only export carries no draft
            # routing promise or stale draft-layer count.
            dspark_source_overlay = None
        elif excluded_mtp_names:
            raise ValueError(
                "refusing a partial DSpark namespace exclusion: the source "
                f"overlay is an atomic three-stage contract, but "
                f"{len(excluded_mtp_names)} of {len(mtp_names)} mtp.* "
                "tensors would be omitted. Exclude the whole `mtp.` "
                "namespace for a body-only artifact or keep all three stages."
            )
    if dspark_cb_sidecar:
        # Discovery above remains the authoritative closed source-layout gate,
        # but the ordinary overlay would re-declare every decoder tensor as
        # source passthrough after we just encoded it.  Quantized DSpark mode
        # owns those targets through CB config groups instead.
        dspark_source_overlay = None
    dspark_body_only = (
        discovered_dspark_source_overlay is not None
        and dspark_source_overlay is None
        and not dspark_cb_sidecar
    )
    if expert_stack_members:
        col_weights = _packed_expert_col_weights(
            col_weights, expert_stack_members, profile)
        print(
            f"[export-cb-stream] collapsed "
            f"{expert_stack_report['members']} per-expert allocation entries "
            f"into {expert_stack_report['stacks']} packed expert stack(s)"
            + (
                f" across {expert_stack_report['format_groups']} format "
                "group(s)"
                if per_expert_config_path is not None else ""
            ),
            flush=True,
        )
    # Every spelling of a routed-expert tensor -> the group prefix that OWNS
    # it: the packed parents both namespaces can name, and every per-expert
    # member under both its checkpoint and its recipe name. This is what lets a
    # CB stack and a delegated per-expert leaf collapse to the same unit id.
    _expert_group_of: dict[str, str] = {}
    for _prefix, _projs in expert_groups.items():
        for _packed_proj in _packed_expert_param_names(profile):
            _packed = f"{_prefix}.{_packed_proj}"
            _expert_group_of[_packed] = _prefix
            _canon_packed = _canonical_qname(_packed, profile)
            if _canon_packed:
                _expert_group_of[_canon_packed] = _prefix
        for _members in _projs.values():
            for _member in _members.values():
                _expert_group_of[_member] = _prefix
                _canon_member = _canonical_qname(_member, profile)
                if _canon_member:
                    _expert_group_of[_canon_member] = _prefix
    for _families in per_expert_plans.values():
        for _entries in _families.values():
            for _entry in _entries:
                if _entry.get("target") is not None:
                    _expert_group_of[str(_entry["target"])] = str(
                        _entry["packed_parent"]
                    ).rsplit(".", 1)[0]

    # Every recipe qname a CB target is VERIFIED and IMATRIX-WEIGHTED under.
    # For a packed stack that is its members, not the stack: the recipe's
    # render identity, the cost rows and the col_weights pickle are all keyed
    # per expert, and inventing a stack-level identity would certify nothing.
    def _identity_scope(qname: str) -> tuple[str, ...]:
        members = expert_stack_members.get(qname)
        if members is None:
            return (qname,)
        return tuple(sorted(members.values()))

    def _packed_stack_target(qname: str) -> bool:
        return qname in expert_stack_members

    _per_expert_plan_by_target = {
        str(entry["target"]): entry
        for families in per_expert_plans.values()
        for entries in families.values()
        for entry in entries
        if entry.get("target") is not None
    }
    _per_expert_source_qnames = {
        str(member)
        for families in per_expert_plans.values()
        for entries in families.values()
        for entry in entries
        if entry.get("source_passthrough")
        for member in entry["members"].values()
    }
    _per_expert_source_prefixes = {
        str(entry["packed_parent"]).rsplit(".", 1)[0]
        for families in per_expert_plans.values()
        for entries in families.values()
        for entry in entries
        if entry.get("source_passthrough")
    }

    def _base_name(qname: str) -> str:
        if dspark_cb_sidecar:
            # Serialized draft tensors always keep their physical checkpoint
            # identity.  Config-group targets are mapped separately below;
            # conflating the two names is the original three-namespace bug.
            if qname == "mtp.0.main_proj":
                return qname
            return dspark_cb_physical_output_for_recipe_target(
                qname, source_config
            )
        plan = _per_expert_plan_by_target.get(qname)
        if plan is not None:
            parent = _export_base_name(
                str(plan["packed_parent"]), profile, skeleton,
                assume_resolvable=True,
            )
            return (
                f"{parent}.{_format_group_slug(plan['format_wire_id'])}"
                if plan.get("discriminated") else parent
            )
        if "." in qname:
            # A packed-stack parent is not a checkpoint tensor, so
            # `_export_base_name` cannot resolve it against the skeleton and
            # falls back to the RECIPE spelling — which, on a wrapped source,
            # gridbook's top-level loader cannot bridge (its rename reuses the
            # model's own hf_to_vllm_mapper, which maps CHECKPOINT prefixes
            # only; a recipe-spelled stack falls through to the arch loader —
            # the documented bug in gridbook moe_toplevel_loader). Name the
            # stack by the checkpoint prefix its OWN expert group carries,
            # like every other tensor in the artifact. Map membership is the
            # guard: only expert-group packed parents have entries, and
            # uniform (no member-plan) groups never appear in
            # expert_stack_members.
            _cp, _leaf = qname.rsplit(".", 1)
            _src_prefix = _canon_to_ckpt_prefix.get(_cp)
            if _src_prefix is not None:
                return f"{_src_prefix}.{_leaf}"
        return _export_base_name(
            qname, profile, skeleton,
            assume_resolvable=_packed_stack_target(qname))

    # --- Stock-CT codecs (mixed container: the plugin delegates non-"scheme"
    # groups to vLLM's CompressedTensors path). REUSE the authoritative
    # export_native_compressed packers — never reimplement packing; RTN only
    # (no GPTQ/act-order), matching how the CB cost stage measures stock rungs. ---
    from prismaquant.format_registry import canonical_format_name
    from prismaquant.export_native_compressed import (
        _quantize_2d as _ct_quantize_2d,
        compute_nvfp4_global_real as _ct_nvfp4_global_real,
    )
    _STOCK_CT_FORMATS = ("NVFP4", "FP8_E4M3")   # FP8_DYNAMIC canonicalizes here

    # --- Classify every target (CB / source passthrough / stock-CT dense /
    # BF16). The passthrough lane splits by WIRE CONTRACT, not by name:
    # `cb_export_config.source_passthrough_wire` says whether a format's scale
    # plane can be normalized into the compressed-tensors namespace or must
    # ship byte-verbatim under the checkpoint's own names. ---
    cb_targets: dict[str, tuple[str, str, int]] = {}
    source_targets: list[str] = []              # CT-normalized (FP8_SOURCE)
    native_source_targets: dict[str, str] = {}  # qname -> byte-verbatim format
    stock_targets: dict[str, str] = {}          # qname -> "NVFP4" | "FP8_E4M3"
    requant_targets: dict[str, str] = {}        # qname -> re-encoded native fmt
    illegal = []
    for qname, fmt in assignment.items():
        if fmt == "BF16":
            continue
        parsed = _parse_cb_format(fmt)
        if parsed is not None:
            cb_targets[qname] = parsed
            continue
        canon = canonical_format_name(fmt)
        if canon in SOURCE_PASSTHROUGH_EXPORT_FORMATS:
            if source_passthrough_wire(canon).ct_normalized:
                source_targets.append(qname)
            else:
                native_source_targets[qname] = canon
            continue
        if canon in _STOCK_CT_FORMATS:
            stock_targets[qname] = canon
            continue
        # Re-quantized native rungs: this producer WROTE the bytes (unlike the
        # passthrough lane) but stock compressed-tensors cannot describe them
        # (unlike the delegated lane), so they get their own emit branch, their
        # own wire id and a scheme-less config group.
        if canon in STREAMING_REQUANT_EXPORT_FORMATS:
            requant_targets[qname] = canon
            continue
        illegal.append((qname, fmt))
    if illegal:
        raise ValueError(
            "streaming CB export carries CB families + stock NVFP4/FP8_DYNAMIC "
            "(CT-delegated) + the source-passthrough family "
            f"{sorted(SOURCE_PASSTHROUGH_EXPORT_FORMATS)} + the re-quantized "
            f"native family {sorted(STREAMING_REQUANT_EXPORT_FORMATS)} + BF16 "
            f"only; unsupported rung(s) {sorted({f for _, f in illegal})} — "
            "assign a legal format or use the in-memory export_nvfp4_cb.")
    if dspark_cb_sidecar:
        expected_native_source_targets = set(dspark_hybrid_source_mapping)
        unexpected_routes = {
            "fp8_source": sorted(source_targets),
            "native_source": sorted(
                qname for qname in native_source_targets
                if qname not in expected_native_source_targets
            ),
            "stock_ct": sorted(stock_targets),
            "requant_native": sorted(requant_targets),
        }
        unexpected_routes = {
            lane: names for lane, names in unexpected_routes.items() if names
        }
        missing_or_wrong_sources = {
            target: native_source_targets.get(target)
            for target in sorted(expected_native_source_targets)
            if native_source_targets.get(target) != _FP8_BLOCK_UE8M0_FORMAT
        }
        if unexpected_routes or missing_or_wrong_sources:
            raise ValueError(
                "DSpark CB sidecar permits exactly 27 physical CB decoder "
                "targets plus four immutable W8A16 source bases (main_proj "
                "and one grouped-BMM wo_a per stage); unexpected routes="
                f"{unexpected_routes}, missing_or_wrong_sources="
                f"{missing_or_wrong_sources}"
            )

    # --- ROUTE-PENDING SHIP GATE. A passthrough rung whose serve route has not
    # been validated stays ON the allocator's menu deliberately: an allocation
    # that wants it is reporting a serving gap worth seeing, and masking the
    # rung would hide that signal while removing the unit's only zero-error
    # option. The fail-closed point is therefore the SHIP step, here. ---
    route_pending = Counter(
        fmt for fmt in list(native_source_targets.values())
        + [canonical_format_name(assignment[q]) for q in source_targets]
        if fmt in ROUTE_PENDING_PASSTHROUGH_FORMATS
    )
    if dspark_source_overlay is not None:
        route_pending.update(
            format_name
            for format_name in dspark_source_overlay.construction_units.values()
            if format_name in ROUTE_PENDING_PASSTHROUGH_FORMATS
        )
    if route_pending and not allow_route_pending_passthrough:
        lanes = ", ".join(
            f"{fmt} -> lane {SOURCE_PASSTHROUGH_CONTRACTS[fmt].serving_route} "
            f"({count} unit(s))"
            for fmt, count in sorted(route_pending.items())
        )
        raise ValueError(
            "refusing to ship a route-pending source passthrough: "
            f"{lanes}. No validated serve route exists for these bytes yet "
            "(allocator_candidates.ROUTE_PENDING_PASSTHROUGH_FORMATS), so the "
            "artifact would load into a lane nothing has been shown to "
            "execute. Pass --allow-route-pending-passthrough "
            "(allow_route_pending_passthrough=True) to ship it anyway; the "
            "acknowledgement is recorded in the artifact's provenance.")

    # Text calibration cannot cover visual/audio sidecar modules that the
    # language-model profile deliberately drops.  Match the resident exporter:
    # delegated stock NVFP4 remains weight-only W4A16 there and is excluded
    # from the static W4A4 scalar contract.
    sidecar_stock = {
        qname
        for qname in stock_targets
        if _canonical_qname(qname, profile) is None
    }

    # Subset gate: every quantised target's export base must live under a
    # declared prefix, else the allocation reaches outside the subset the caller
    # asked to export (a mistake worth failing on, not silently over/under
    # covering). Passthrough is filtered by the same prefixes below.
    if subset_prefixes is not None:
        outside = sorted(
            q for q in list(cb_targets) + list(source_targets)
            + list(native_source_targets) + list(stock_targets)
            if not any(_base_name(q).startswith(p)
                       for p in subset_prefixes))
        if outside:
            raise ValueError(
                f"--subset-prefix {subset_prefixes}: {len(outside)} allocation "
                f"target(s) resolve outside the subset, e.g. {outside[:5]} — "
                "the layer_config and the declared subset disagree")

    def _resolve_target(qname, suffix=".weight"):
        """Locate a target's source: a stacked skeleton tensor, an expert
        group (per-expert on disk), or a resolved skeleton key. Returns
        (kind, handle)."""
        plan = _per_expert_plan_by_target.get(qname)
        if plan is not None:
            parent = str(plan["packed_parent"])
            prefix, packed_proj = parent.rsplit(".", 1)
            grp = expert_groups.get(prefix)
            if grp is None:
                raise KeyError(
                    f"{qname}: per-expert format group has no source prefix "
                    f"{prefix!r}"
                )
            return "experts", (prefix, packed_proj, grp)
        key = _try_resolve_skeleton(qname, skeleton, profile, suffix)
        if key is not None:
            return "tensor", key
        # Packed-expert groups are keyed by CHECKPOINT prefixes. Resolve the
        # recipe qname directly and through the profile's source mapping, and
        # accept only packed parents declared by that profile.
        candidates = [qname]
        if profile is not None:
            try:
                mapped = profile.source_tensor_name(qname)
                if mapped not in candidates:
                    candidates.append(mapped)
            except Exception:
                pass
        packed_names = _packed_expert_param_names(profile)
        for candidate in candidates:
            if "." not in candidate:
                continue
            prefix, packed_proj = candidate.rsplit(".", 1)
            if not prefix.endswith(".experts") or packed_proj not in packed_names:
                continue
            grp = expert_groups.get(prefix)
            if grp is not None:
                return "experts", (prefix, packed_proj, grp)
        return None, None

    def _target_shape(qname):
        kind, h = _resolve_target(qname)
        if kind == "tensor":
            return tuple(skeleton.logical_shape(h))
        if kind == "experts":
            prefix, packed_proj, grp = h
            projections = _packed_expert_projection_names(profile, packed_proj)
            member_plan = expert_stack_members.get(qname)
            n = (
                len({expert_id for _projection, expert_id in member_plan})
                if member_plan is not None
                else _n_experts(grp, projections)
            )
            shapes = [skeleton.logical_shape(grp[p][0] + ".weight")
                      for p in projections]
            in_f = int(shapes[0][1])
            if any(len(s) != 2 or int(s[1]) != in_f for s in shapes):
                raise ValueError(
                    f"{qname}: incompatible expert projection shapes {shapes}")
            return (n, sum(int(s[0]) for s in shapes), in_f)
        raise KeyError(f"{qname}: no streaming source (tensor or expert group)")

    # --- Coverage gate (lazy: shapes only). ---
    for qname, (grid, mode, k) in cb_targets.items():
        shape = _target_shape(qname)
        in_f = int(shape[-1])
        if in_f % cb.SUPERBLOCK != 0:
            raise ValueError(
                f"{qname}: in_features={in_f} not a multiple of "
                f"{cb.SUPERBLOCK}")
        if qname not in col_weights:
            raise ValueError(
                f"{qname}: CB target has no col_weights entry (no silent RTN)")
        cwn = col_weights[qname].numel()
        n_exp = int(shape[0]) if len(shape) == 3 else 1
        if cwn not in (in_f, n_exp * in_f):
            raise ValueError(
                f"{qname}: col_weights has {cwn} elements but the weight "
                f"wants {in_f} or {n_exp}x{in_f}")

    # --- Stock-CT coverage + expert-stack gate. Stock rungs stream for DENSE
    # (2-D) Linears only; a MoE expert stack assigned a stock format has no
    # safe streaming pack here (the CB container's stock config emits a
    # packed-name regex vLLM's MoE dispatch cannot match to its per-expert
    # probes, and the CT codec is 2-D), so fail fast pointing at the fix. ---
    stock_expert: list[str] = []
    for qname in stock_targets:
        kind, _h = _resolve_target(qname)
        if kind is None:
            raise KeyError(
                f"{qname}: assigned {stock_targets[qname]} but no streaming "
                "source (tried the .weight key + the profile-mapped checkpoint "
                "name)")
        shape = _target_shape(qname)
        if kind == "experts" or len(shape) == 3:
            stock_expert.append(qname)
            continue
        if stock_targets[qname] == "NVFP4" and int(shape[-1]) % 16 != 0:
            raise ValueError(
                f"{qname}: stock NVFP4 needs in_features % 16 == 0 (group 16), "
                f"got in_features={int(shape[-1])}")
    if stock_expert:
        raise ValueError(
            "streaming CB export carries stock NVFP4/FP8_DYNAMIC on DENSE "
            "Linears only; these MoE expert-stack target(s) were assigned a "
            f"stock format: {sorted(stock_expert)[:5]}"
            f"{' ...' if len(stock_expert) > 5 else ''} "
            f"({len(stock_expert)} total). Assign expert stacks a CB rung "
            "(nvfp4_cb / fp8_cb), FP8_SOURCE, or BF16 — or use the in-memory "
            "export_nvfp4_cb on a model small enough to materialise. The dense "
            "tier is where vanilla NVFP4/FP8_DYNAMIC won the A/B; constrain the "
            "allocator to keep experts on CB/passthrough.")

    # --- Quantized token embeddings (`quantized_embedding` declaration).
    # Mirrors export_nvfp4_cb:767-830 exactly; see that block for the full
    # rationale. The short version: a third class of stock target, packed by
    # the same CT codec but claimed by GridBook's embedding method rather than
    # by a config group (vLLM's compressed-tensors embedding path RAISES for
    # FP8/NVFP4, so a config group naming it refuses to load), and WEIGHT-ONLY
    # (a lookup has no input activation and the serving method registers no
    # `input_global_scale`, so an emitted one is an unmatched checkpoint key).
    #
    # This lane needs it because a 13.0 GB card budget cannot afford a bf16
    # embedding: on Qwen3.8-27B that is 2.543 GB, ~20% of the artifact, and
    # NVFP4 buys it back for 0.715 GB at a measured 0.001063 KL.
    #
    # Shapes come from `_target_shape` (safetensors header only) rather than
    # the resident tensor the in-memory exporter can afford to hold. ---
    def _declared_vocab_size() -> int:
        # Multimodal checkpoints keep the LM's vocab under `text_config`; a
        # wrapper config with no top-level vocab_size would otherwise read as
        # zero and disable the shape half of the cross-check silently.
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        for holder in (cfg, cfg.get("text_config") or {},
                       cfg.get("language_config") or {}):
            if isinstance(holder, dict) and holder.get("vocab_size"):
                return int(holder["vocab_size"])
        return 0

    _vocab_rows = _declared_vocab_size()

    def _is_embedding_name(q: str) -> bool:
        return q == "model.embed_tokens" or q.endswith(".embed_tokens")

    embedding_stock: dict[str, str] = {}
    for _q, _f in stock_targets.items():
        _shape = _target_shape(_q)
        _rows = int(_shape[0]) if len(_shape) == 2 else -1
        _named = _is_embedding_name(_q)
        _shaped = _vocab_rows > 0 and _rows == _vocab_rows and not (
            _q == "lm_head" or _q.endswith(".lm_head"))
        if _named != _shaped:
            raise ValueError(
                f"{_q}: cannot classify as a token embedding — the name says "
                f"{_named} but the checkpoint shape says {_shaped} "
                f"(rows={_rows}, vocab_size={_vocab_rows}). An embedding is "
                "served by GridBook's lookup method and a Linear by a config "
                "group; the two dispatches are not interchangeable, so this "
                "refuses rather than guessing.")
        if _named:
            embedding_stock[_q] = _f
    if embedding_stock:
        # Read the record back through the consumer's rules before any bytes
        # are written: an unroutable format or an lm_head slipped into the
        # recipe must fail the export, not the load.
        build_quantized_embedding_declaration(embedding_stock)
        # An embedding is not a sidecar tower: it keeps its `model.` prefix in
        # vLLM's module tree and is claimed by the declaration, not by a
        # weight-only config group. Both memberships would otherwise fire on a
        # profile whose LM mapping does not name the embedding.
        sidecar_stock -= set(embedding_stock)

    # --- Stock NVFP4 fused-sibling coherence (mirrors export_nvfp4_cb /
    # export_native_compressed): q/k/v and gate/up landing on NVFP4 MUST share
    # ONE weight_global_scale or vLLM's fused loader sees inconsistent
    # per-tensor globals. Take the max over each fused group's natural
    # global_real (streamed — one weight resident at a time) and override every
    # sibling's pack. Singleton groups get their own global, exactly like the
    # in-memory exporter (so the streamed bytes are byte-identical). ---
    _nvfp4_shared_global: dict[str, torch.Tensor] = {}
    _nvfp4_groups: dict[str, list[str]] = {}
    for _q, _f in stock_targets.items():
        if _f != "NVFP4":
            continue
        _gk = (profile.fused_sibling_group(_q)
               if profile is not None else None) or _q
        _nvfp4_groups.setdefault(_gk, []).append(_q)
    for _members in _nvfp4_groups.values():
        _grs = []
        for _m in _members:
            _k, _h = _resolve_target(_m)
            _w = skeleton.dequant_weight(_h).to(device)
            _grs.append(_ct_nvfp4_global_real(_w, 16).reshape(()))
            del _w
        _shared = torch.stack(_grs).max()
        for _m in _members:
            _nvfp4_shared_global[_m] = _shared

    fp4_activation_targets = {
        qname
        for qname, (grid, _mode, _k) in cb_targets.items()
        if grid == "fp4"
    } | {
        qname
        for qname, fmt in stock_targets.items()
        if fmt == "NVFP4" and qname not in sidecar_stock
        and qname not in embedding_stock
    }
    activation_execution_contract = None
    activation_scales_by_physical_target: dict[str, float] = {}

    # --- Resolve codebooks. Production learned cells are immutable bundle
    # inputs trained before cost; the bounded export-time trainer survives only
    # for an explicitly unstamped historical research run. ---
    provided = spec.get("codebooks", {}) if source == "learned" else {}
    train = bool(spec.get("train", False))
    iters = int(spec.get("iters", 4))
    seed = int(spec.get("seed", 0))
    train_cap = int(spec.get("train_cap", 1 << 20))
    learned_bundle = None
    if _scoped_bundle_export:
        from prismaquant.cb_learned_bundle import load_bundle_cached

        learned_bundle = load_bundle_cached(_env_cb_context.codebook_bundle_path)
    codebooks: dict[tuple[str, str], object] = {}
    target_cb: dict[str, tuple] = {}
    by_group: dict[tuple[str, str], list[str]] = {}
    expert_role_plans: dict[
        str, tuple[RoutedMoECodebookRole, ...]
    ] = {}
    cb_group_target_names: dict[tuple[str, str], tuple[str, ...]] = {}
    role_group_keys: set[tuple[str, str]] = set()
    # Recipe qname -> pooled stack cell, for routed stacks whose books were
    # burned per (layer, stack, rung).
    pooled_stack_cells: dict[str, str] = {}
    for qname in cb_targets:
        fmt = assignment[qname]
        grid, mode, k = cb_targets[qname]
        source_kind = (
            codebook_source_for_format(fmt, _env_cb_context)
            if _scoped_bundle_export else source
        )
        target_shape = _target_shape(qname)
        target_kind, target_handle = _resolve_target(qname)
        if source_kind == "learned":
            from prismaquant.cb_learned_bundle import refuse_routed_moe_learned

            refuse_routed_moe_learned(
                qname,
                routed_moe=(target_kind == "experts" or len(target_shape) == 3),
            )
        plan = _per_expert_plan_by_target.get(qname)
        if (
            source_kind == "learned"
            and _scoped_bundle_export
            and (target_kind == "experts" or len(target_shape) == 3)
        ):
            if learned_bundle is None:
                raise AssertionError("scoped routed CBL has no learned bundle")
            if (
                grid != "fp8"
                or mode != "product"
                or k not in ROUTED_MOE_CBL_BANK_RUNGS
            ):
                raise ValueError(
                    f"{qname}/{fmt}: routed learned books are banked only for "
                    "FP8-CB K28--K33"
                )
            members = expert_stack_members.get(qname)
            if target_kind != "experts" or members is None:
                raise ValueError(
                    f"{qname}: learned expert export requires the "
                    "per-expert source/member map so gate, up, and down input "
                    "identities can be verified independently"
                )
            prefix, packed_proj, group = target_handle
            projections = _packed_expert_projection_names(profile, packed_proj)
            expected = (
                ("gate_proj", "up_proj")
                if str(packed_proj) == "gate_up_proj"
                else ("down_proj",)
            )
            if tuple(projections) != expected:
                raise ValueError(
                    f"{qname}: Gridbook routed expert ABI expects {expected}, "
                    f"profile declared {tuple(projections)}"
                )
            physical_target = _base_name(qname)
            # Campaign rule R1: how the BOOKS were burned decides the spelling,
            # so read the bundle's own record rather than guessing here. A
            # stack-keyed cell pools gate and up, so the fused weight names ONE
            # codebook on the packed target -- the same spelling a lattice
            # layer uses. Role-keyed cells keep the per-half declaration below.
            stack_qname = bundle_stack_qname(qname)
            if (
                learned_bundle.has_cell(stack_qname, fmt)
                and learned_bundle.routed_book_keying(stack_qname, fmt)
                == ROUTED_BOOK_KEYING_STACK
            ):
                codebook = learned_bundle.codebook_for(stack_qname, fmt)
                group_key = (stack_qname, fmt)
                if group_key in codebooks:
                    raise ValueError(
                        f"duplicate routed learned stack group {group_key}"
                    )
                codebooks[group_key] = codebook
                by_group[group_key] = [qname]
                role_group_keys.add(group_key)
                pooled_stack_cells[qname] = stack_qname
                target_cb[qname] = (stack_qname, fmt, codebook, "learned")
                # No per-member `validate_inputs` here, unlike the role branch
                # below: the pooled cell's recorded imatrix identity is the
                # packed entry materialized per expert, and the exporter holds
                # the raw entry. What binds this target instead is the recipe's
                # own render identity on the fused source weight
                # (`validate_cb_render_source_weight`) plus the codebook digest
                # cross-check against the immutable bundle.
                continue
            roles: list[RoutedMoECodebookRole] = []
            for projection in projections:
                # The immutable cell is shared by every expert subgroup at
                # this layer/projection/rung.  Only the runtime target retains
                # the split-stack discriminator.
                logical_qname = bundle_role_qname(qname, projection)
                logical_target = logical_role_qname(
                    physical_target, projection
                )
                role_cw, role_members = stacked_role_col_weights(
                    packed_qname=qname,
                    projection=projection,
                    member_qnames=members,
                    col_weights=col_weights,
                )
                first_expert = min(group[projection])
                source_shape = skeleton.logical_shape(
                    group[projection][first_expert] + ".weight"
                )
                if len(source_shape) != 2:
                    raise ValueError(
                        f"{logical_qname}: expert member must be rank 2, got "
                        f"{source_shape}"
                    )
                codebook = learned_bundle.codebook_for(logical_qname, fmt)
                group_key = (logical_qname, fmt)
                if group_key in codebooks:
                    raise ValueError(
                        f"duplicate routed learned role group {group_key}"
                    )
                codebooks[group_key] = codebook
                by_group[group_key] = [qname]
                role_group_keys.add(group_key)
                cb_group_target_names[group_key] = (logical_target,)
                roles.append(RoutedMoECodebookRole(
                    projection=projection,
                    qname=logical_qname,
                    ref=logical_qname,
                    format_name=fmt,
                    codebook=codebook,
                    col_weights=role_cw,
                    output_rows=int(source_shape[0]),
                    member_qnames=role_members,
                ))
            expert_role_plans[qname] = tuple(roles)
            first_role = roles[0]
            target_cb[qname] = (
                first_role.ref,
                fmt,
                first_role.codebook,
                "learned",
            )
            continue
        # Every group in a declared mixed layer owns one physical sidecar.
        # This includes a family that happens to have a single format while
        # its sibling family is split: the per-expert footprint contract
        # charges once per declared sub-stack, so sharing that group's lattice
        # ref with a dense target would make the emitted bytes disagree.
        # Uniform layers have no retained plan and preserve the legacy shared
        # ref byte-for-byte.
        if plan is not None:
            ref = (
                f"pe_l{plan['layer']}_{plan['family']}_"
                f"{_format_group_slug(plan['format_wire_id'])}"
            )
        else:
            ref = (
                qname
                if _scoped_bundle_export and source_kind == "learned"
                else (_role_of(qname) if source_kind == "learned" else "lattice")
            )
        by_group.setdefault((ref, fmt), []).append(qname)

    # --- SPLIT-BOOK SHIP GATE (campaign rule R1). The predicate is structural
    # and producer-side: count the distinct codebooks one fused routed weight's
    # scheme would name. Nothing here asserts what a runtime does; the runtime
    # consequence appears only in the human-facing message.
    books_by_fused_target: dict[str, dict[str, tuple[str, ...]]] = {}
    for routed_qname, routed_roles in expert_role_plans.items():
        if len(routed_roles) < 2:
            continue
        books_by_fused_target[_base_name(routed_qname)] = {
            # One ref names one immutable bundle cell, and the digest
            # cross-check below binds one value to each ref, so counting refs
            # counts codebooks.
            role.projection: role.ref
            for role in routed_roles
        }
    split_book_targets = fused_targets_with_split_books(books_by_fused_target)
    if split_book_targets and not allow_per_role_books:
        raise ValueError(describe_split_book_refusal(split_book_targets))
    if split_book_targets:
        print(
            "[export-cb-stream] --allow-per-role-books: shipping "
            f"{len(split_book_targets)} fused routed weight(s) with per-role "
            "books; the acknowledgement is stamped on the shipcard",
            flush=True,
        )

    for (ref, fmt), qnames in by_group.items():
        if (ref, fmt) in role_group_keys:
            continue
        grid, mode, k = cb_targets[qnames[0]]
        source_kind = (
            codebook_source_for_format(fmt, _env_cb_context)
            if _scoped_bundle_export else source
        )
        if source_kind == "lattice":
            codebooks[(ref, fmt)] = cb._resolve_codebook(
                k, grid, mode, None, torch.device(device))
            kind = "lattice"
        elif _scoped_bundle_export:
            if len(qnames) != 1 or learned_bundle is None:
                raise AssertionError(
                    f"{ref}/{fmt}: learned bundle cell must own one qname"
                )
            qname = qnames[0]
            target_kind, target_handle = _resolve_target(qname)
            if target_kind != "tensor":
                from prismaquant.cb_learned_bundle import refuse_routed_moe_learned

                refuse_routed_moe_learned(qname, routed_moe=True)
            weight = skeleton.dequant_weight(target_handle).to(device)
            codebooks[(ref, fmt)] = learned_bundle.codebook_for(
                qname,
                fmt,
                weight=weight,
                col_weights=col_weights[qname],
            )
            kind = "learned"
        elif train:
            codebooks[(ref, fmt)] = _train_shared_codebook_streaming(
                skeleton, profile, expert_groups, _resolve_target,
                qnames, col_weights, grid=grid, mode=mode, k=k, seed=seed,
                iters=iters, train_cap=train_cap, device=device,
                members_by_target=expert_stack_members)
            kind = "learned"
        elif ref in provided:
            codebooks[(ref, fmt)] = provided[ref]
            kind = "learned"
        else:
            raise ValueError(
                f"role {ref!r} ({fmt}): learned but no codebook + train=False")
        for q in qnames:
            target_cb[q] = (ref, fmt, codebooks[(ref, fmt)], kind)

    materialized_codebook_tensors = {
        name: tensor
        for (ref, fmt), codebook in codebooks.items()
        for name, tensor in _codebook_tensors(ref, fmt, codebook).items()
    }
    selected_codebook_digests = {
        name: hashlib.sha256(
            tensor.to(torch.float16).cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for name, tensor in materialized_codebook_tensors.items()
    }
    materialized_codebook_digests = (
        dict(_env_cb_context.codebook_content_digests or {})
        if _scoped_bundle_export else {}
    )
    for name, digest in selected_codebook_digests.items():
        previous = materialized_codebook_digests.get(name)
        if previous is not None and previous != digest:
            raise ValueError(
                f"export_nvfp4_cb_streaming: selected codebook {name!r} "
                "differs from the immutable bundle digest"
            )
        materialized_codebook_digests[name] = digest
    selected_refs_by_format: dict[str, dict[str, tuple[str, ...]]] = {}
    for qname, (ref, fmt, codebook, _kind) in target_cb.items():
        roles = expert_role_plans.get(qname)
        if roles is None:
            # A pooled routed stack reports under its bundle cell name, which
            # is the packed parent without any format-subgroup discriminator --
            # the same name the bundle's own ref map uses.
            selected_refs_by_format[pooled_stack_cells.get(qname, qname)] = {
                fmt: _codebook_tensor_names(ref, fmt, codebook)
            }
            continue
        for role in roles:
            selected_refs_by_format[role.qname] = {
                role.format_name: _codebook_tensor_names(
                    role.ref, role.format_name, role.codebook
                )
            }
    if _scoped_bundle_export:
        refs_by_format = {
            str(qname): dict(by_format)
            for qname, by_format in (
                _env_cb_context.codebook_refs_by_qname_format or {}
            ).items()
        }
        for qname, by_format in selected_refs_by_format.items():
            target_formats = refs_by_format.setdefault(qname, {})
            for fmt, refs in by_format.items():
                previous = target_formats.get(fmt)
                if previous is not None and tuple(
                    (previous,) if isinstance(previous, str) else previous
                ) != tuple(refs):
                    raise ValueError(
                        f"{qname}/{fmt}: streaming exporter refs differ from "
                        "immutable bundle refs"
                    )
                target_formats[fmt] = refs
    else:
        refs_by_format = None
    serialization_context = CBSerializationContext(
        scale_coding=scale_coding,
        codebook_source=source,
        codebook_source_scope=(
            _env_cb_context.codebook_source_scope
            if _scoped_bundle_export else None
        ),
        scale_sweep=(
            _env_cb_context.scale_sweep
            if _scoped_bundle_export else bool(scale_sweep)
        ),
        scale_sweep_scope=(
            _env_cb_context.scale_sweep_scope
            if _scoped_bundle_export else None
        ),
        ldlq=_env_cb_context.ldlq,
        ldlq_scope=getattr(_env_cb_context, "ldlq_scope", "all" if _env_cb_context.ldlq else "none"),
        minchain=_env_cb_context.minchain,
        minchain_version=_env_cb_context.minchain_version,
        encode_tier=_env_cb_context.encode_tier,
        activation_contract=_claimed_activation_contract,
        activation_execution=(
            NVFP4_ACTIVATION_EXECUTION
            if _claimed_activation_contract is not None
            else None
        ),
        codebook_refs=None if _scoped_bundle_export else {
            qname: _codebook_tensor_names(ref, fmt, codebook)
            for qname, (ref, fmt, codebook, _kind) in target_cb.items()
            if qname not in expert_role_plans
        },
        codebook_refs_by_qname_format=refs_by_format,
        codebook_content_digests=materialized_codebook_digests,
        codebook_bundle_path=(
            _env_cb_context.codebook_bundle_path
            if _scoped_bundle_export else None
        ),
    )
    validate_cb_serialization_context_stamp(
        _recipe_cb_context_stamp,
        serialization_context,
        where="export_nvfp4_cb_streaming",
    )

    # --- ROUTE-STATUS GATE (campaign rule R3, principle 9). ----------------
    # Serving eligibility is judged per artifact, at export. This is the first
    # point where every input the verdict needs exists -- crucially
    # `expert_role_plans`, which is what makes a stack's per-role codebook
    # split visible -- and it is still before any byte is written, so a refusal
    # costs nothing. The defect it closes: the shipped DSv4 body's 11 routed
    # FP8-CB layers bind per-role learned books, Gridbook's persistent-B lane
    # refuses those, and nothing on the producer side consumed that fact; a
    # user found it at serve time.
    from .cb_route_status_gate import gate_cb_export_units

    cb_route_status_provenance = gate_cb_export_units(
        assignment=assignment,
        quantized_targets=(*cb_targets, *stock_targets),
        routed_units=expert_stack_members,
        role_split_units=(
            qname for qname, roles in expert_role_plans.items() if roles
        ),
        shape_of=_target_shape,
        allow_unbacked_route=allow_unbacked_route,
        non_native_target=non_native_target,
        exporter="export_nvfp4_cb_streaming",
    )
    from prismaquant.nvfp4_cb_footprint import _ldlq_for_format

    routed_ldlq = sorted(
        qname
        for qname in (set(expert_role_plans) | set(pooled_stack_cells))
        if _ldlq_for_format(assignment[qname], serialization_context)
    )
    if routed_ldlq:
        raise ValueError(
            "routed learned CBL reuses the immutable no-LDLQ burn "
            "identity; PRISMAQUANT_CB_LDLQ_SCOPE must exclude FP8 for "
            f"{routed_ldlq[:5]}"
        )

    _ldlq_telemetry_qnames = {
        telemetry_qname
        for qname in cb_targets
        if _ldlq_for_format(assignment[qname], serialization_context)
        for telemetry_qname in (
            tuple(role.qname for role in expert_role_plans[qname])
            if qname in expert_role_plans
            else (pooled_stack_cells.get(qname, qname),)
        )
    }
    ldlq_telemetry = None
    if _ldlq_telemetry_qnames:
        from prismaquant.cb_ldlq_gate_telemetry import (
            LDLQGateTelemetryCollector,
        )

        ldlq_telemetry = LDLQGateTelemetryCollector(
            expected_qnames=_ldlq_telemetry_qnames,
            kernel_stamp=cb.canonical_ldlq_kernel_stamp(),
        )
    ldlq_activation_loader = None
    if serialization_context.ldlq:
        if activation_cache_dir is None:
            raise ValueError(
                "export_nvfp4_cb_streaming: LDLQ requires activation_cache_dir"
            )
        from prismaquant.cb_ldlq import CBLDLQActivationLoader

        ldlq_activation_loader = CBLDLQActivationLoader(
            activation_cache_dir,
            model_dir=model_dir,
            profile=profile,
            expert_stack_members=expert_stack_members,
            replay_device=device,
        )
    cb_render_source_collector = None
    if _dspark_render_recipe is not None and not dspark_cb_sidecar:
        raise ValueError(
            "DSpark render recipe stamp is valid only with --dspark-cb-sidecar"
        )
    if (
        dspark_cb_sidecar
        and not allow_unstamped_research
        and _dspark_render_recipe is None
    ):
        raise ValueError(
            "production DSpark CB sidecar export requires the immutable "
            "dspark_render_recipe stamp emitted by "
            "build_dsv4_dspark_cb_sidecar_inputs.py"
        )
    if (
        cb_targets
        and dspark_cb_sidecar
        and _dspark_render_recipe is not None
    ):
        from prismaquant.production_weight_cache import (
            CBRenderSourceIdentityCollector,
            validate_cb_render_identity_metadata,
        )

        validate_cb_render_identity_metadata(
            _recipe_cb_render_identity,
            expected_context=serialization_context,
            expected_formats_by_qname={
                member: (assignment[qname],)
                for qname in sorted(cb_targets)
                for member in _identity_scope(qname)
            },
            col_weights=col_weights,
            require_source_complete=False,
            require_minchain_cells=serialization_context.minchain,
            where="export_nvfp4_cb_streaming DSpark render identity seed",
        )
        _validate_dspark_streaming_render_recipe(
            _dspark_render_recipe,
            render_identity=_recipe_cb_render_identity,
            source_model_identity=source_model_identity,
            model_dir=model_dir,
            skeleton=skeleton,
            assignment=_recipe_assignment,
        )
        cb_render_source_collector = CBRenderSourceIdentityCollector(
            _recipe_cb_render_identity,
            where="export_nvfp4_cb_streaming DSpark decoded source",
        )
    elif cb_targets and _recipe_cb_render_identity is not None:
        from prismaquant.production_weight_cache import (
            validate_cb_render_identity_metadata,
        )

        validate_cb_render_identity_metadata(
            _recipe_cb_render_identity,
            expected_context=serialization_context,
            expected_formats_by_qname={
                member: (assignment[qname],)
                for qname in sorted(cb_targets)
                for member in _identity_scope(qname)
            },
            col_weights=col_weights,
            require_minchain_cells=serialization_context.minchain,
            where="export_nvfp4_cb_streaming assignment render identity",
        )
    elif (
        cb_targets
        and production_recipe_stamped
        and _research_cost_selection is None
    ):
        raise ValueError(
            "export_nvfp4_cb_streaming: stamped production CB assignment is "
            "missing its value-bearing render identity"
        )
    elif cb_targets and not (
        allow_unstamped_research or _research_cost_selection is not None
    ):
        raise ValueError(
            "export_nvfp4_cb_streaming: CB export requires a value-bearing "
            "render identity; pass allow_unstamped_research=True only for "
            "an explicit non-production experiment"
        )
    elif _recipe_cb_render_identity is not None:
        raise ValueError(
            "export_nvfp4_cb_streaming: non-CB assignment carries a stale "
            "CB render identity"
        )

    warm_session = None
    if warm_state_dir is not None:
        from prismaquant.cb_warm_state import (
            CBWarmStartSession,
            CBWarmStateStore,
        )

        warm_store = CBWarmStateStore(warm_state_dir)
        warm_records = {}
        identity = (
            _recipe_cb_render_identity
            if isinstance(_recipe_cb_render_identity, dict)
            else {}
        )
        source_shapes = identity.get("source_weights_shapes", {})
        source_digests = identity.get(
            "source_weights_content_sha256", {}
        )
        col_shapes = identity.get("col_weights_shapes", {})
        col_digests = identity.get("col_weights_content_sha256", {})
        for qname in sorted(cb_targets):
            # Collapsed per-expert stacks deliberately cold-fallback: their
            # export imatrix pools member vectors, so no individual member's
            # selected scale is the stack's scale-search argmin.
            if _identity_scope(qname) != (qname,):
                continue
            if not all(
                qname in values
                for values in (
                    source_shapes, source_digests, col_shapes, col_digests
                )
            ):
                continue
            record = warm_store.load_matching(
                qname=qname,
                format_name=assignment[qname],
                source_shape=list(source_shapes[qname]),
                source_digest=str(source_digests[qname]),
                col_weights_shape=list(col_shapes[qname]),
                col_weights_digest=str(col_digests[qname]),
                context=serialization_context,
            )
            if record is not None:
                warm_records[qname] = record
        warm_session = CBWarmStartSession(
            warm_records,
            all_qnames=sorted(cb_targets),
            verify_sample=int(warm_verify_sample),
        )
        print(
            f"[export-cb-stream] encoder warm state: "
            f"{len(warm_records)}/{len(cb_targets)} matching; "
            f"verifying {len(warm_session.verify_qnames)}",
            flush=True,
        )

    # Validate persisted render/serialization identity before reading the
    # activation cache.  Stale recipes must fail for the primary cause and
    # must not trigger an expensive calibration replay.
    if _claimed_activation_contract is not None and fp4_activation_targets:
        if activation_cache_dir is None:
            raise ValueError(
                "export_nvfp4_cb_streaming: production FP4 activation "
                "contract requires activation_cache_dir; refusing "
                "uncalibrated fused W4A4"
            )
        activation_scale_policy_id = resolve_input_global_scale_policy(
            activation_scale_policy
        )
        from prismaquant.moe_imatrix import (
            synthesize_packed_expert_activation_samples,
        )

        packed_candidates = {
            qname for qname in fp4_activation_targets
            if qname.endswith((".gate_up_proj", ".down_proj"))
        }
        # A checkpoint whose experts are PACKED caches only the experts
        # module's input, so the routed intermediate has to be replayed from
        # the checkpoint. A checkpoint whose experts are PER-EXPERT cached both
        # stages' real inputs, so the same numbers are pooled, not replayed —
        # and the replay's own entry condition would silently decline it.
        per_expert_stacks = {
            qname: expert_stack_members[qname]
            for qname in packed_candidates
            if qname in expert_stack_members
        }
        supplemental_max_abs: dict[str, float] = {}
        supplemental_samples: dict[str, object] = {}
        if per_expert_stacks:
            from prismaquant.moe_imatrix import (
                per_expert_stage_activation_calibration,
            )

            supplemental_samples, supplemental_max_abs = (
                per_expert_stage_activation_calibration(
                    activation_cache_dir,
                    per_expert_stacks,
                    policy=activation_scale_policy_id,
                )
            )
        replay_candidates = packed_candidates - set(per_expert_stacks)
        if replay_candidates and profile is not None:
            supplemental_samples.update(
                synthesize_packed_expert_activation_samples(
                    model_dir,
                    activation_cache_dir,
                    replay_candidates,
                    profile,
                    device=device,
                )
            )
        (
            logical_scales,
            activation_calibration_sources,
        ) = calibrated_input_global_scales_with_sources(
            fp4_activation_targets,
            activation_cache_dir=activation_cache_dir,
            policy=activation_scale_policy_id,
            profile=profile,
            supplemental_activations=supplemental_samples,
            supplemental_max_abs=supplemental_max_abs,
            calibration_device=device,
        )
        (
            activation_execution_contract,
            activation_scales_by_physical_target,
        ) = build_execution_contract(
            logical_scales,
            policy=activation_scale_policy_id,
            target_name=_base_name,
            calibration_sources=activation_calibration_sources,
            profile=profile,
        )
    verified_cb_source_qnames: set[str] = set()

    # DELTA-EXPORT: a CB group's codebook is a byte-copy input, so compare this
    # run's serialized codebook against the prior sidecar ONCE per (ref, fmt).
    # A group whose codebook differs makes every target on it re-encode.
    group_cb_ok: dict[tuple[str, str], bool] = {}
    if prior is not None:
        for (ref, fmt), codebook in codebooks.items():
            cur_t = _codebook_tensors(ref, fmt, codebook)
            ok = True
            for tname, t in cur_t.items():
                pt = prior.codebook_tensor(tname)
                if pt is None or tuple(pt.shape) != tuple(t.shape) \
                        or not torch.equal(pt, t):
                    ok = False
                    break
            group_cb_ok[(ref, fmt)] = ok

    # --- Build the streaming plan + config in one metadata pass. ---
    writer = _StreamWriter()
    counts: Counter[str] = Counter()
    ignore: list[str] = []
    cb_targets_set = set(cb_targets)
    source_set = set(source_targets)
    native_source_set = set(native_source_targets)
    stock_set = set(stock_targets)
    emitted_bases: set[str] = set()   # checkpoint tensor keys we consume
    cb_serialized_shapes: dict[str, tuple[int, ...]] = {}
    cb_output_tensor_names: set[str] = set()
    planned_cb_tensor_bytes = 0
    # Physical tensor names each routed-expert target contributes, so the
    # source_passthrough declaration and the route reconciliation describe what
    # was actually planned rather than what the naming rules imply.
    tensor_names_by_target: dict[str, list[str]] = {}

    # CB + FP8_SOURCE targets (keyed by canonical/recipe qname).
    for qname in list(cb_targets) + list(source_targets):
        export_base = _base_name(qname)
        kind, h = _resolve_target(qname)
        if qname in source_set:
            wkey = _try_resolve_skeleton(qname, skeleton, profile)
            scale_entry = skeleton._fp8_scale_inv_map.get(qname + ".weight")
            skey = scale_entry[1] if scale_entry is not None else None
            if wkey is None or skey not in skeleton or \
                    skeleton.get_dtype(wkey) != torch.float8_e4m3fn:
                raise ValueError(
                    f"{qname}: FP8_SOURCE but source is not native fp8 with "
                    "a profile-resolved serialized scale")
            emitted_bases.add(wkey)
            emitted_bases.add(skey)
            wsh = skeleton.get_shape(wkey)
            ssh = skeleton.get_shape(skey)
            writer.add(export_base + ".weight", torch.float8_e4m3fn, wsh,
                       (lambda k=wkey: skeleton.load(k).contiguous()),
                       reader=(lambda k=wkey: _pin_prefetched_tensor(
                           skeleton.load(k), device)),
                       encoder=(lambda value: value.contiguous()))
            writer.add(
                export_base + ".weight_scale", torch.float32, ssh,
                (lambda k=skey: skeleton.load(k).to(torch.float32)
                 .contiguous()),
                reader=(lambda k=skey: _pin_prefetched_tensor(
                    skeleton.load(k), device)),
                encoder=(lambda value: value.to(torch.float32).contiguous()),
            )
            counts[canonical_format_name(assignment[qname])] += 1
            continue
        grid, mode, k = cb_targets[qname]
        ref, fmt, codebook, _ = target_cb[qname]
        shape = _target_shape(qname)
        rows = 1
        for d in shape[:-1]:
            rows *= int(d)
        n_sb = int(shape[-1]) // cb.SUPERBLOCK
        coding = scale_coding if grid == "fp4" else cb.SCALE_CODING_V1
        ts = cb.nvfp4_cb_type_size(k, grid, coding)
        if kind == "tensor":
            emitted_bases.add(h)
        packed_shape = ((shape[0], shape[1], n_sb * ts) if len(shape) == 3
                        else (rows, n_sb * ts))
        payload = cb_tensor_payload_breakdown(
            fmt,
            shape,
            qname=qname,
            context=serialization_context,
        )
        planned_packed_bytes = _nbytes(torch.uint8, packed_shape)
        if planned_packed_bytes != payload["packed_weight_bytes"]:
            raise AssertionError(
                f"{qname}: streaming cb_qweight plan is "
                f"{planned_packed_bytes}B, accounting expected "
                f"{payload['packed_weight_bytes']}B"
            )
        state: dict = {}

        def _pack(qname=qname, h=(kind, h), grid=grid, mode=mode, k=k,
                  codebook=codebook, coding=coding, shape=shape,
                  packed_shape=packed_shape, state=state):
            from prismaquant.nvfp4_cb_footprint import _ldlq_for_format

            ldlq_for_this = _ldlq_for_format(assignment[qname], serialization_context)
            packed, scale = _stream_pack_target(
                skeleton, profile, h, qname, grid, mode, k, codebook,
                col_weights[qname], scale_sweep_for_format(
                    assignment[qname], serialization_context
                ), coding, shape, device,
                serialization_context.encode_tier,
                _recipe_cb_render_identity,
                cb_render_source_collector,
                verified_cb_source_qnames,
                expert_stack_members.get(qname),
                expert_roles=expert_role_plans.get(qname),
                learned_bundle=learned_bundle,
                all_col_weights=col_weights,
                warm_session=warm_session,
                format_name=assignment[qname],
                ldlq_activation_loader=ldlq_activation_loader if ldlq_for_this else None,
                ldlq_telemetry=ldlq_telemetry if ldlq_for_this else None,
            )
            state["scale"] = scale
            return packed.reshape(packed_shape)

        qw_name = export_base + ".cb_qweight"
        scale_shape = tuple(int(d) for d in shape[:-1])
        scale_name = export_base + ".weight_scale"
        input_scale_name = export_base + ".input_global_scale"
        planned_scale_bytes = (
            _nbytes(torch.float32, scale_shape) if grid == "fp8" else 0
        )
        if planned_scale_bytes != payload["fp8_row_scale_bytes"]:
            raise AssertionError(
                f"{qname}: streaming row-scale plan is "
                f"{planned_scale_bytes}B, accounting expected "
                f"{payload['fp8_row_scale_bytes']}B"
            )
        planned_input_scale_bytes = (
            _nbytes(torch.float32, (1,))
            if grid == "fp4" and _claimed_activation_contract is not None
            else 0
        )
        if planned_input_scale_bytes != payload["input_global_scale_bytes"]:
            raise AssertionError(
                f"{qname}: streaming input-global-scale plan is "
                f"{planned_input_scale_bytes}B, accounting expected "
                f"{payload['input_global_scale_bytes']}B"
            )
        planned_tensor_bytes = (
            planned_packed_bytes
            + planned_scale_bytes
            + planned_input_scale_bytes
        )
        if planned_tensor_bytes != payload["tensor_payload_bytes"]:
            raise AssertionError(
                f"{qname}: streaming CB tensor plan is "
                f"{planned_tensor_bytes}B, accounting expected "
                f"{payload['tensor_payload_bytes']}B"
            )
        cb_serialized_shapes[qname] = tuple(int(dim) for dim in shape)
        planned_cb_tensor_bytes += planned_tensor_bytes
        # Planned output tensors (name, dtype, shape) — the eligibility gate.
        expected = [(qw_name, torch.uint8, packed_shape)]
        cb_output_tensor_names.add(qw_name)
        if grid == "fp8":
            expected.append((scale_name, torch.float32, scale_shape))
            cb_output_tensor_names.add(scale_name)
        elif _claimed_activation_contract is not None:
            expected.append((input_scale_name, torch.float32, (1,)))
            cb_output_tensor_names.add(input_scale_name)
        tensor_names_by_target[qname] = [name for name, _d, _s in expected]

        # DELTA-EXPORT eligibility: same format + scheme signature + byte-equal
        # codebook + every planned output already present in the prior at the
        # planned dtype+shape => byte-copy instead of re-encode.
        reason = "disabled"
        if prior is not None:
            cur_subset = cb_scheme_reuse_signature(build_cb_scheme(
                ref=ref,
                fmt=fmt,
                grid=grid,
                mode=mode,
                k=k,
                codebook=codebook,
                scale_coding=coding,
                activation_contract=(
                    NVFP4_ACTIVATION_CONTRACT_KEY
                    if _claimed_activation_contract is not None
                    else None
                ),
            ))
            reason = _cb_reuse_reason(
                prior, export_base, fmt, cur_subset, expected,
                group_cb_ok.get((ref, fmt), False))

        if prior is not None and reason is None:
            for name, dtype, _sh in expected:
                writer.add(name, dtype, prior.shape(name), None,
                           copy_src=prior.raw_slice(name))
            reuse["copied"] += 1

            def _fresh(qname=qname, h=(kind, h), grid=grid, mode=mode, k=k,
                       codebook=codebook, coding=coding, shape=shape,
                       packed_shape=packed_shape, scale_shape=scale_shape,
                       qw_name=qw_name, scale_name=scale_name,
                       input_scale_name=input_scale_name,
                       export_base=export_base):
                from prismaquant.nvfp4_cb_footprint import _ldlq_for_format

                ldlq_for_this = _ldlq_for_format(assignment[qname], serialization_context)
                packed, scale = _stream_pack_target(
                    skeleton, profile, h, qname, grid, mode, k, codebook,
                    col_weights[qname], scale_sweep_for_format(
                        assignment[qname], serialization_context
                    ), coding, shape, device,
                    serialization_context.encode_tier,
                    _recipe_cb_render_identity,
                    cb_render_source_collector,
                    verified_cb_source_qnames,
                    expert_stack_members.get(qname),
                    expert_roles=expert_role_plans.get(qname),
                    learned_bundle=learned_bundle,
                    all_col_weights=col_weights,
                    warm_session=warm_session,
                    format_name=assignment[qname],
                    ldlq_activation_loader=ldlq_activation_loader if ldlq_for_this else None,
                    ldlq_telemetry=ldlq_telemetry if ldlq_for_this else None,
                )
                out = {qw_name: packed.reshape(packed_shape)}
                if scale is not None:
                    out[scale_name] = scale.reshape(scale_shape).to(
                        torch.float32).contiguous()
                if grid == "fp4" and _claimed_activation_contract is not None:
                    out[input_scale_name] = input_global_scale_tensor(
                        activation_scales_by_physical_target[export_base]
                    )
                return out
            reuse["verify_pool"].append(
                {"base": export_base, "specs": expected, "fresh": _fresh})
        else:
            if kind == "tensor":
                def _read_cb(h=h):
                    return _prefetch_source_weight(skeleton, h, device)

                def _encode_cb(
                    weight,
                    qname=qname,
                    grid=grid,
                    mode=mode,
                    k=k,
                    codebook=codebook,
                    coding=coding,
                    packed_shape=packed_shape,
                    state=state,
                ):
                    from prismaquant.nvfp4_cb_footprint import _ldlq_for_format

                    ldlq_for_dense = _ldlq_for_format(assignment[qname], serialization_context)
                    packed, scale = _encode_prefetched_cb_tensor(
                        weight,
                        qname=qname,
                        grid=grid,
                        mode=mode,
                        k=k,
                        codebook=codebook,
                        cw=col_weights[qname],
                        scale_sweep=scale_sweep_for_format(
                            assignment[qname], serialization_context
                        ),
                        coding=coding,
                        device=device,
                        encode_tier=serialization_context.encode_tier,
                        cb_render_identity=_recipe_cb_render_identity,
                        cb_render_source_collector=cb_render_source_collector,
                        verified_source_qnames=verified_cb_source_qnames,
                        warm_session=warm_session,
                        format_name=assignment[qname],
                        ldlq_activation_loader=ldlq_activation_loader if ldlq_for_dense else None,
                        ldlq_telemetry=ldlq_telemetry if ldlq_for_dense else None,
                    )
                    state["scale"] = scale
                    return packed.reshape(packed_shape)

                writer.add(
                    qw_name,
                    torch.uint8,
                    packed_shape,
                    _pack,
                    reader=_read_cb,
                    encoder=_encode_cb,
                )
            else:
                # Expert stacks retain their existing one-device-buffer
                # producer. Prefetching an entire next stack into pinned host
                # memory would duplicate a 10GB-class working set on the GB10
                # unified pool and violate the bounded-residency contract.
                writer.add(qw_name, torch.uint8, packed_shape, _pack)
            if grid == "fp8":
                def _scale(state=state, scale_shape=scale_shape):
                    return state["scale"].reshape(scale_shape).to(
                        torch.float32).contiguous()
                writer.add(scale_name, torch.float32, scale_shape, _scale)
            elif _claimed_activation_contract is not None:
                if export_base not in activation_scales_by_physical_target:
                    raise AssertionError(
                        f"{qname}: claimed FP4 activation contract has no "
                        f"physical scalar for {export_base!r}"
                    )
                writer.add(
                    input_scale_name,
                    torch.float32,
                    (1,),
                    lambda value=activation_scales_by_physical_target[
                        export_base
                    ]: input_global_scale_tensor(value),
                )
            if prior is not None:
                reuse["encoded"] += 1
                reuse["reasons"][reason] += 1
        counts[fmt] += 1

    # BYTE-VERBATIM source passthrough: the checkpoint's own element plane and
    # its own scale plane, copied BYTE FOR BYTE under their CHECKPOINT names.
    # Covers every format in `DELEGATED_NATIVE_PASSTHROUGH_FORMATS` — DSv4's
    # nibble-packed MXFP4 routed experts and its UE8M0 block-FP8 body are the
    # same wire contract at two element grids, so they are one branch.
    #
    # Three things this branch deliberately does NOT do, each differing from the
    # CT-normalized FP8_SOURCE branch above for a stated reason:
    #
    #  * it does not widen the scale to F32. FP8_SOURCE renames
    #    `.weight_scale_inv` -> `.weight_scale` and casts, because vLLM's stock
    #    block-FP8 CT path reads an fp32 scale. Nothing reads an E8M0 plane
    #    except the loader that wrote it, which wants the exponents it shipped;
    #    casting would quadruple 12.5 GB of DSv4 scale bytes to buy a format no
    #    consumer asked for. That is the whole reason these formats are separate
    #    registry entries rather than FP8_SOURCE with a different scale dtype.
    #  * it does not rename. The native DeepseekV4 loader resolves
    #    `layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}`; emitting the live
    #    `model.layers.N.mlp.experts.…` spelling would produce an artifact whose
    #    tensors nothing resolves.
    #  * it does not add the target to `ignore`. `ignore` means "unquantized,
    #    load it as-is"; these ARE quantized and are described by their own
    #    config group — the same rule the FP8_SOURCE branch follows.
    #
    # The copy runs through `_LazySkeleton.raw_slice` + `_StreamWriter`'s
    # chunked copy path, so no source tensor is ever materialised.
    from prismaquant.cb_source_decode import checkpoint_weight_to_live_name

    for qname in sorted(native_source_targets):
        fmt = native_source_targets[qname]
        spec = _fr_get_format(fmt)
        wkey = _try_resolve_skeleton(qname, skeleton, profile)
        if wkey is None:
            raise ValueError(
                f"{qname}: assigned {fmt} but no source tensor (tried the "
                ".weight key + the profile-mapped checkpoint name). The "
                "source-passthrough family is passthrough-only — never "
                "synthesize it.")
        live_weight = checkpoint_weight_to_live_name(wkey, profile=profile)
        scale_entry = skeleton._fp8_scale_inv_map.get(live_weight)
        skey = scale_entry[1] if scale_entry is not None else None
        if skey is None or skey not in skeleton:
            raise ValueError(
                f"{qname}: assigned {fmt} but {wkey!r} has no resolved "
                f"{spec.scale_dtype_name} scale sibling; an element plane "
                "without its scale plane is not a loadable tensor.")
        elements_per_byte = _PASSTHROUGH_ELEMENTS_PER_BYTE[
            str(spec.weight_element_dtype)]
        if elements_per_byte > 1:
            # A SUB-BYTE element plane is not self-describing: its stored dtype
            # is just "one byte". Only the checkpoint's own declaration
            # (`autoscale.declared_fp4_expert_dtype`, surfaced as the scale
            # map's `mxfp4_names`) says those bytes are packed nibbles, and the
            # passthrough copies bytes it never interprets — so without the
            # declaration it would happily ship reinterpreted garbage. A
            # byte-per-element plane needs no such gate: its dtype IS the claim.
            declared = getattr(
                skeleton._fp8_scale_inv_map, "mxfp4_names", frozenset())
            if live_weight not in declared:
                raise ValueError(
                    f"{qname}: assigned {fmt} but the checkpoint does not "
                    f"DECLARE {wkey!r} as a packed sub-byte plane (config "
                    "expert_dtype). A shape heuristic here would ship "
                    "reinterpreted garbage.")
        wsh = tuple(int(d) for d in skeleton.get_shape(wkey))
        ssh = tuple(int(d) for d in skeleton.get_shape(skey))
        wdtype = skeleton.get_dtype(wkey)
        sdtype = skeleton.get_dtype(skey)
        _assert_passthrough_planes_agree(
            qname, fmt, spec, wkey, wsh, wdtype, skey, ssh, sdtype)
        emitted_bases.add(wkey)
        emitted_bases.add(skey)
        writer.add(wkey, wdtype, wsh, None, copy_src=skeleton.raw_slice(wkey))
        writer.add(skey, sdtype, ssh, None, copy_src=skeleton.raw_slice(skey))
        tensor_names_by_target[qname] = [wkey, skey]
        counts[fmt] += 1

    # Stock-CT DENSE targets: analytic on-disk tensors packed RTN via the
    # export_native_compressed codec (byte-identical to export_nvfp4_cb). ONE
    # producer packs the weight once and caches every suffix tensor; the writer
    # streams them one at a time. RTN is deterministic, so RESUME re-runs the
    # group from its base boundary and rewrites identical bytes.
    for qname in sorted(stock_targets):
        canon_fmt = stock_targets[qname]
        export_base = _base_name(qname)
        kind, h = _resolve_target(qname)              # dense: kind == "tensor"
        shape = _target_shape(qname)
        override = (_nvfp4_shared_global.get(qname)
                    if canon_fmt == "NVFP4" else None)
        emitted_bases.add(h)
        state: dict = {}

        def _render(h=h, canon_fmt=canon_fmt, override=override, state=state,
                    export_base=export_base):
            if "out" not in state:
                w = skeleton.dequant_weight(h).to(device)
                packed = _ct_quantize_2d(
                    w,
                    canon_fmt,
                    nvfp4_global_real_override=override,
                    input_global_scale_override=(
                        activation_scales_by_physical_target.get(export_base)
                        if canon_fmt == "NVFP4"
                        else None
                    ),
                )
                state["out"] = {s: t.cpu().contiguous()
                                for s, t in packed.items()}
                del w
            return state["out"]

        def _render_prefetched(
            w,
            canon_fmt=canon_fmt,
            override=override,
            state=state,
            export_base=export_base,
        ):
            if "out" not in state:
                w = w.to(
                    device,
                    non_blocking=bool(
                        w.device.type == "cpu" and w.is_pinned()
                    ),
                )
                packed = _ct_quantize_2d(
                    w,
                    canon_fmt,
                    nvfp4_global_real_override=override,
                    input_global_scale_override=(
                        activation_scales_by_physical_target.get(export_base)
                        if canon_fmt == "NVFP4"
                        else None
                    ),
                )
                state["out"] = {
                    suffix: tensor.cpu().contiguous()
                    for suffix, tensor in packed.items()
                }
                del w
            return state["out"]

        specs = [
            spec
            for spec in _stock_output_specs(canon_fmt, shape)
            if not (
                (qname in sidecar_stock or qname in embedding_stock)
                and "input" in spec[0]
            )
        ]
        expected = [(export_base + "." + s, d, o) for s, d, o in specs]
        # DELTA-EXPORT: RTN stock rungs are deterministic from the (unchanged)
        # source weight. FP8_E4M3 is per-channel (no cross-tensor coupling);
        # NVFP4's only cross-tensor input is the fused-group shared global,
        # which the union-find coherence invariant pins identical whenever this
        # target is on NVFP4 in both allocations (q/k/v, gate/up move as a unit,
        # weights unchanged). So the prior having every planned output at the
        # exact dtype+shape is a sound copy gate.
        stock_ok = prior is not None and all(
            prior.matches_dtype_shape(n, d, o) for n, d, o in expected)
        if stock_ok:
            for name, dtype, _sh in expected:
                writer.add(name, dtype, prior.shape(name), None,
                           copy_src=prior.raw_slice(name))
            reuse["copied"] += 1
        else:
            for output_index, ((name, dtype, out_shape),
                               (suffix, _d, _o)) in enumerate(zip(
                                   expected, specs)):
                def _prod(suffix=suffix, _render=_render):
                    return _render()[suffix]
                if output_index == 0:
                    writer.add(
                        name,
                        dtype,
                        out_shape,
                        _prod,
                        reader=(lambda h=h: _prefetch_source_weight(
                            skeleton, h, device)),
                        encoder=(lambda weight, suffix=suffix,
                                 render=_render_prefetched:
                                 render(weight)[suffix]),
                    )
                else:
                    writer.add(name, dtype, out_shape, _prod)
            if prior is not None:
                reuse["encoded"] += 1
                reuse["reasons"][
                    "stock_not_in_prior" if not prior.has(expected[0][0])
                    else "stock_dtype_shape_mismatch"] += 1
        counts[canon_fmt] += 1

    # Re-quantized native DENSE targets (MXFP8_UE8M0_G32). Structurally the
    # stock-CT loop above with a different packer: one producer encodes the
    # weight once into its element + scale planes and the writer streams them
    # one at a time. The codec is RTN with no search and no cross-tensor input,
    # so it is deterministic from the (unchanged) source weight and the same
    # dtype+shape copy gate the stock lane uses is sound here too.
    for qname in sorted(requant_targets):
        canon_fmt = requant_targets[qname]
        export_base = _base_name(qname)
        kind, h = _resolve_target(qname)              # dense: kind == "tensor"
        shape = _target_shape(qname)
        emitted_bases.add(h)
        state: dict = {}

        def _render(h=h, canon_fmt=canon_fmt, state=state):
            if "out" not in state:
                w = skeleton.dequant_weight(h).to(device)
                packed = _requant_pack(canon_fmt, w)
                state["out"] = {s: t.cpu().contiguous()
                                for s, t in packed.items()}
                del w
            return state["out"]

        def _render_prefetched(w, canon_fmt=canon_fmt, state=state):
            if "out" not in state:
                w = w.to(
                    device,
                    non_blocking=bool(
                        w.device.type == "cpu" and w.is_pinned()
                    ),
                )
                packed = _requant_pack(canon_fmt, w)
                state["out"] = {
                    suffix: tensor.cpu().contiguous()
                    for suffix, tensor in packed.items()
                }
                del w
            return state["out"]

        specs = _requant_output_specs(canon_fmt, shape)
        expected = [(export_base + "." + s, d, o) for s, d, o in specs]
        requant_ok = prior is not None and all(
            prior.matches_dtype_shape(n, d, o) for n, d, o in expected)
        if requant_ok:
            for name, dtype, _sh in expected:
                writer.add(name, dtype, prior.shape(name), None,
                           copy_src=prior.raw_slice(name))
            reuse["copied"] += 1
        else:
            for output_index, ((name, dtype, out_shape),
                               (suffix, _d, _o)) in enumerate(zip(
                                   expected, specs)):
                def _prod(suffix=suffix, _render=_render):
                    return _render()[suffix]
                if output_index == 0:
                    writer.add(
                        name,
                        dtype,
                        out_shape,
                        _prod,
                        reader=(lambda h=h: _prefetch_source_weight(
                            skeleton, h, device)),
                        encoder=(lambda weight, suffix=suffix,
                                 render=_render_prefetched:
                                 render(weight)[suffix]),
                    )
                else:
                    writer.add(name, dtype, out_shape, _prod)
            if prior is not None:
                reuse["encoded"] += 1
                reuse["reasons"][
                    "requant_not_in_prior" if not prior.has(expected[0][0])
                    else "requant_dtype_shape_mismatch"] += 1
        tensor_names_by_target[qname] = [n for n, _d, _o in expected]
        counts[canon_fmt] += 1

    # Passthrough: every remaining checkpoint tensor verbatim (BF16/norms/etc).
    # Per-expert tensors consumed by a stacked CB target are NOT passthrough.
    # Expert groups are keyed by the on-disk (checkpoint) prefix; a nested
    # source (Qwen3.5-VLM `model.language_model.*`, DSv4) needs the CANONICAL
    # prefix for the membership test against the recipe-named CB targets —
    # without it every per-expert bf16 source ships verbatim NEXT TO its
    # packed CB stack (35B first-contact: 31511 copied tensors, 82 GB
    # artifact at a 4.75 bpp target).
    consumed_expert_bases = set()
    for prefix, projs in expert_groups.items():
        # Legacy/direct packed assignments have no member plan. Consume every
        # source projection belonging to an actually-CB packed parent. A
        # partial layer (gate_up=CB, down=BF16) must retain the untouched
        # expert tensors for its BF16 parent.
        for packed_proj in _packed_expert_param_names(profile):
            checkpoint_qname = f"{prefix}.{packed_proj}"
            canon_qname = _canonical_qname(checkpoint_qname, profile)
            variants = {checkpoint_qname}
            if canon_qname is not None:
                variants.add(canon_qname)
            if not variants & cb_targets_set:
                continue
            for projection in _packed_expert_projection_names(
                profile, packed_proj
            ):
                for base in projs.get(projection, {}).values():
                    consumed_expert_bases.add(base + ".weight")
    # A discriminated sub-stack is not named by its packed parent, so consume
    # only its selected expert ids through the explicit member plan.
    for target, member_qnames in expert_stack_members.items():
        if target not in cb_targets_set:
            continue
        for projection, expert_id in member_qnames:
            member = member_qnames[(projection, expert_id)]
            # The group key is in the recipe namespace; the source handle is
            # the profile-planned checkpoint base.
            prefix = member.rsplit(".", 2)[0]
            source_group = expert_groups[prefix]
            consumed_expert_bases.add(
                source_group[projection][expert_id] + ".weight"
            )
    resolved_source_scale_keys = {
        entry[1] for entry in skeleton._fp8_scale_inv_map.values()
    }
    _verbatim_prefix_reader = getattr(
        profile, "source_passthrough_prefixes", None
    )
    source_verbatim_prefixes = (
        tuple(_verbatim_prefix_reader())
        if callable(_verbatim_prefix_reader)
        else ()
    )
    # FLOOR block-FP8: units no allocation target claimed. They ship weight AND
    # scale and get DECLARED, instead of losing their scale to the skip below
    # and being cast to bf16 under an `ignore` entry. See
    # `_floor_block_fp8_units` for the failure this closes.
    floor_fp8_units, floor_fp8_scale_owner = _floor_block_fp8_units(
        skeleton,
        emitted_bases=emitted_bases,
        consumed_expert_bases=consumed_expert_bases,
        claimed_qnames=(cb_targets_set | source_set | native_source_set
                        | stock_set),
        profile=profile,
        subset_prefixes=subset_prefixes,
        excluded_namespaces=excluded_namespaces,
    )
    for name in skeleton.keys():
        if subset_prefixes is not None and \
                not any(name.startswith(p) for p in subset_prefixes):
            continue   # outside the declared subset (e.g. non-MTP body layers)
        if any(name.startswith(p) for p in excluded_namespaces):
            # OMITTED ENTIRELY: no tensor, no index entry, no `ignore` line, no
            # declaration. The prefix test is on the raw checkpoint key, so a
            # unit's scale and other companions leave with it automatically
            # rather than by a second rule that could disagree.
            counts["excluded"] += 1
            continue
        if name in emitted_bases or name in consumed_expert_bases:
            continue
        if name not in floor_fp8_scale_owner and \
                name in resolved_source_scale_keys:
            # A resolved scale map says how a weight CAN be decoded; it does
            # not say this export actually consumed that weight.  Once MTP's
            # physical scale pairs became visible, the old global skip dropped
            # every verbatim ``mtp.*.scale`` plane while still copying its
            # weight.  Keep a remaining scale only when the profile explicitly
            # owns the sibling weight through its verbatim namespace; an
            # otherwise unallocated source unit still follows the established
            # drop/ignore path unless the floor-FP8 declaration above claimed
            # it.
            weight_key = (
                name[: -len(".scale")] + ".weight"
                if name.endswith(".scale") else None
            )
            if (
                weight_key in emitted_bases
                or weight_key in consumed_expert_bases
                or weight_key is None
                or not any(
                    weight_key.startswith(prefix)
                    for prefix in source_verbatim_prefixes
                )
            ):
                continue
        if name.endswith(".weight_scale_inv"):
            continue   # legacy FP8_SOURCE companions remain target-owned
        if name.endswith(".weight"):
            ckpt_qname = name[:-len(".weight")]
        elif _try_resolve_direct_packed_expert(
                name, skeleton, profile) == name:
            ckpt_qname = name
        else:
            ckpt_qname = None
        canon = _canonical_qname(ckpt_qname, profile) if ckpt_qname else None
        # The byte-verbatim passthrough lane is in this test for the same
        # reason FP8_SOURCE is: its tensors were already emitted (under their
        # checkpoint names, so `emitted_bases` catches them too) and, more
        # importantly, they must NOT land in `ignore`. A per-expert passthrough
        # group is not collapsed, so every one of its 768 per-layer bases
        # passes through this loop.
        if canon in cb_targets_set or canon in source_set \
                or canon in native_source_set or canon in stock_set:
            continue
        shape = skeleton.get_shape(name)
        dtype = skeleton.get_dtype(name)
        writer.add(
            name,
            dtype,
            shape,
            (lambda k=name: skeleton.load(k).contiguous()),
            reader=(lambda k=name: _pin_prefetched_tensor(
                skeleton.load(k), device)),
            encoder=(lambda value: value.contiguous()),
        )
        # A declared floor block-FP8 unit must NOT land in `ignore`: the two
        # statements contradict each other, and `ignore` is the one that
        # silently loses the scale.  Its scale plane rides this same loop (it
        # is exempted from the skip above) and needs no entry of its own.
        if ckpt_qname is not None and ckpt_qname in floor_fp8_units:
            counts["floor_fp8_declared"] += 1
        elif ckpt_qname is not None and len(shape) >= 2:
            ignore.append(ckpt_qname)
        counts["copied"] += 1

    # --- Codebook sidecar (small; in-memory) + config + write. ---
    cb_tensor_blobs: dict[str, torch.Tensor] = dict(
        materialized_codebook_tensors
    )
    codebook_file = "cb_codebooks.pqcb" if cb_tensor_blobs else None
    if set(cb_serialized_shapes) != set(cb_targets):
        missing = sorted(set(cb_targets) - set(cb_serialized_shapes))
        extra = sorted(set(cb_serialized_shapes) - set(cb_targets))
        raise AssertionError(
            "streaming CB payload coverage does not match assignment: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if activation_execution_contract is not None:
        planned_names = set(writer.names())
        emitted_scale_targets = {
            target
            for target in activation_scales_by_physical_target
            if target + ".input_global_scale" in planned_names
        }
        if emitted_scale_targets != set(activation_scales_by_physical_target):
            raise AssertionError(
                "streaming NVFP4 activation scalar coverage differs from the "
                "claimed execution contract: missing="
                f"{sorted(set(activation_scales_by_physical_target) - emitted_scale_targets)[:8]}"
            )
    serialized_payload = cb_assignment_payload_breakdown(
        {qname: assignment[qname] for qname in cb_targets},
        cb_serialized_shapes,
        context=serialization_context,
    )
    if _recipe_cb_context_stamp is not None or _recipe_cb_tensor_stamps:
        # A collapsed stack's stamps are its MEMBERS' — the recipe priced 768
        # per-expert tensors, so that is the scope its per-layer identities
        # have to be checked against. The packed plan is then reconciled to
        # that scope explicitly rather than being compared to a stamp set the
        # recipe never wrote.
        recipe_formats = {
            member: assignment[qname]
            for qname in cb_targets
            for member in _identity_scope(qname)
        }
        recipe_shapes = dict(cb_serialized_shapes)
        for qname in cb_targets:
            member_shapes = _member_serialized_shapes(
                qname, expert_stack_members.get(qname), expert_groups,
                skeleton, profile)
            if member_shapes:
                recipe_shapes.pop(qname, None)
                recipe_shapes.update(member_shapes)
        validate_cb_assignment_serialization_stamps(
            recipe_formats,
            recipe_shapes,
            context=serialization_context,
            stamps=_recipe_cb_tensor_stamps,
            where="export_nvfp4_cb_streaming",
        )
        _assert_packed_plan_reconciles_to_recipe(
            recipe_formats,
            recipe_shapes,
            serialized_payload,
            expert_stack_members,
            context=serialization_context,
            where="export_nvfp4_cb_streaming",
        )
    if planned_cb_tensor_bytes != serialized_payload["tensor_payload_bytes"]:
        raise AssertionError(
            f"streaming CB plan is {planned_cb_tensor_bytes}B, accounting "
            f"expected {serialized_payload['tensor_payload_bytes']}B"
        )
    validate_cb_sidecar_tensors(
        serialized_payload,
        cb_tensor_blobs,
        where="export_nvfp4_cb_streaming",
    )
    serialized_payload_summary = cb_payload_summary(serialized_payload)

    def _cb_target_name(qname: str) -> str:
        physical = _base_name(qname)
        if dspark_cb_sidecar:
            return dspark_cb_construction_target_for_physical_output(
                physical, source_config
            )
        return physical

    def _delegated_target_name(qname: str) -> str:
        # MEASURED on the serving stack (vLLM 0.27.1 + gridbook 0.8.11,
        # wrapped Qwen3.5 sources): find_matched_target matches quant-config
        # targets against the LANGUAGE-TOWER-RELATIVE module path, so
        # canonical `model.layers.*` spellings match and full live-tree
        # spellings (`language_model.model.*`) leave the module unquantized —
        # which gridbook then refuses fail-closed at weight load. Keep the
        # profile's internal renames but strip the wrapper prefix.
        name = (
            profile.to_vllm_internal_name(qname)
            if profile is not None
            else qname
        )
        if name.startswith("language_model."):
            name = name[len("language_model."):]
        return name

    def _decision_unit_id(qname: str) -> str:
        """The unit id the declaration names this target by.

        A UNIT is what the allocator decided atomically. For anything inside a
        routed-expert group that is the experts MODULE, not the individual
        Linear: the CB route names a packed parent (``…experts.gate_up_proj``)
        while the delegated route names 768 per-expert leaves, and the two have
        to collapse to the same id or "is this unit claimed twice?" cannot be
        asked. Everything else — a dense body Linear on the UE8M0 lane — is its
        own unit.
        """
        unit = _expert_group_of.get(qname, qname)
        if dspark_cb_sidecar and unit in native_source_targets:
            construction = dspark_construction_unit_for_physical_target(
                unit,
                num_hidden_layers=(
                    discovered_dspark_source_overlay.num_hidden_layers
                ),
                n_mtp_layers=(
                    discovered_dspark_source_overlay.n_mtp_layers
                ),
            )
            if construction is None:
                raise ValueError(
                    f"{unit}: DSpark native source unit has no construction "
                    "namespace mapping"
                )
            return construction
        return unit

    def _physical_expert_module(prefix: str) -> str | None:
        """The SERIALIZED prefix K0.2 names this routed-expert group by.

        The activation attestation is keyed by physical names
        (``layers.7.ffn.experts``) while units are recipe names
        (``model.layers.7.mlp.experts``); reconciling the two needs this bridge
        and not a string heuristic. ``assume_resolvable`` is required because
        the packed parent never exists on disk — the same reason
        ``_base_name`` needs it for a collapsed CB stack.
        """
        for packed_proj in sorted(_packed_expert_param_names(profile)):
            base = _export_base_name(
                f"{prefix}.{packed_proj}", profile, skeleton,
                assume_resolvable=True)
            if "." in base:
                return base.rsplit(".", 1)[0]
        return None

    def _source_passthrough_units() -> dict[str, str]:
        """``{unit id: registry format}`` for every DELEGATED-NATIVE unit.

        No exhaustiveness claim: this says which units are delegated, not that
        every unit in the model was enumerated. What it does enforce is that a
        unit is delegated WHOLE — a routed-expert group with some members
        delegated and some not would ship half a layer to the model's own
        loader and half to gridbook's codec, which neither can serve.

        Re-quantized native rungs are included: the consumer's dispatcher reads
        this ONE map to decide "native route or CB decoder" and refuses a unit
        whose format id it does not know, so a unit omitted here would be read
        as CB — the one wrong answer that loads. They are still not
        byte-verbatim, and the config group's ``weights.source_passthrough``
        flag is what carries that distinction per unit.
        """
        units: dict[str, str] = {}
        for qname, fmt in requant_targets.items():
            units[_decision_unit_id(qname)] = fmt
        for qname, fmt in native_source_targets.items():
            if qname in _per_expert_source_qnames:
                # PROPOSED v1 per-expert declaration is the sole authority for
                # a mixed MXFP4_SOURCE subgroup.  Adding the layer-level unit
                # here would double-declare it and erase the expert partition.
                continue
            unit = _decision_unit_id(qname)
            previous = units.setdefault(unit, fmt)
            if previous != fmt:
                raise ValueError(
                    f"{unit}: source-passthrough unit mixes {previous} and "
                    f"{fmt}; a unit ships on ONE contract or the export cannot "
                    "declare it")
        for prefix, projs in expert_groups.items():
            if prefix in _per_expert_source_prefixes:
                continue
            if prefix not in units:
                continue
            members = {member for proj in projs.values()
                       for member in proj.values()}
            delegated = {
                member for member in members
                if member in native_source_set
                or (_canonical_qname(member, profile) or member)
                in native_source_set
            }
            if delegated != members:
                raise ValueError(
                    f"{prefix}: {len(delegated)} of {len(members)} per-expert "
                    "tensors are on a source passthrough; a delegated expert "
                    "group is served whole by the model's own loader or not "
                    "at all")
        return units

    def _route_reconciliation_sets(units):
        """Collect the sets ``assert_routes_reconcile`` compares.

        Kept beside the export because only here are BOTH namespaces in hand:
        the declaration speaks recipe names and the config groups speak
        serialized ones.
        """
        cb_units = {_decision_unit_id(qname) for qname in cb_targets}
        cb_modules: set[str] = set()
        passthrough_modules: set[str] = set()
        for prefix in expert_groups:
            physical = _physical_expert_module(prefix)
            if physical is None:
                continue
            if prefix in units:
                passthrough_modules.add(physical)
            elif prefix in cb_units:
                cb_modules.add(physical)
        return {
            "cb_units": cb_units,
            "passthrough_units": set(units),
            "cb_tensors": {
                name for qname in cb_targets
                for name in tensor_names_by_target.get(qname, ())
            },
            "passthrough_tensors": {
                name
                for qname in (*native_source_targets, *requant_targets)
                if qname not in _per_expert_source_qnames
                for name in tensor_names_by_target.get(qname, ())
            },
            "cb_modules": cb_modules,
            "passthrough_modules": passthrough_modules,
            "attested": set(routed_moe_attested_module_names(
                activation_execution_contract)),
        }

    def _per_expert_wire_declaration_and_payload():
        """Materialize PROPOSED v1 plus its independently checked byte sum."""

        if not per_expert_plans:
            return None, None
        entry_bytes = {
            entry.name: entry.nbytes
            for entry in writer._entries
        }
        declaration_layers: dict[str, dict[str, list[dict[str, object]]]] = {}
        payload_groups: dict[str, dict[str, object]] = {}
        for layer, families in sorted(
            per_expert_plans.items(), key=lambda item: int(item[0])
        ):
            if set(families) != {"w13", "w2"}:
                raise ValueError(
                    f"layer {layer}: {PER_EXPERT_FORMAT_GROUPS_KEY} requires "
                    f"both w13 and w2 families, got {sorted(families)}"
                )
            layer_record: dict[str, list[dict[str, object]]] = {}
            for family in ("w13", "w2"):
                records = []
                for entry in sorted(
                    families[family],
                    key=lambda item: str(item["format_wire_id"]),
                ):
                    if entry["source_passthrough"]:
                        prefix = str(entry["packed_parent"]).rsplit(".", 1)[0]
                        tensor_prefix = _physical_expert_module(prefix)
                        if tensor_prefix is None:
                            raise ValueError(
                                f"layer {layer} {family}: cannot map source "
                                "expert prefix into the artifact namespace"
                            )
                        tensor_names = sorted({
                            name
                            for member in entry["members"].values()
                            for name in tensor_names_by_target.get(member, ())
                        })
                        codebook_bytes = 0
                        codebook_tensor_names: list[str] = []
                    else:
                        target = str(entry["target"])
                        tensor_prefix = _base_name(target)
                        tensor_names = sorted(
                            tensor_names_by_target.get(target, ())
                        )
                        per_tensor = serialized_payload["per_tensor"][target]
                        codebook_bytes = int(
                            per_tensor["sidecar_payload_bytes"]
                        )
                        sidecar_identities = per_tensor.get(
                            "sidecar_identities",
                            (per_tensor["sidecar_identity"],),
                        )
                        codebook_tensor_names = sorted({
                            str(name)
                            for sidecar in sidecar_identities
                            for name in sidecar["codebook_ref"]
                        })
                    missing_names = sorted(
                        name for name in tensor_names if name not in entry_bytes
                    )
                    if missing_names:
                        raise AssertionError(
                            f"layer {layer} {family} "
                            f"{entry['format_wire_id']}: planned declaration "
                            f"names absent tensors {missing_names[:8]}"
                        )
                    tensor_bytes = sum(entry_bytes[name] for name in tensor_names)
                    record = {
                        "format_wire_id": str(entry["format_wire_id"]),
                        "expert_ids": [int(value) for value in entry["expert_ids"]],
                        "tensor_prefix": tensor_prefix,
                    }
                    records.append(record)
                    payload_groups[
                        f"{layer}/{family}/{entry['format_wire_id']}"
                    ] = {
                        **record,
                        "tensor_names": tensor_names,
                        "codebook_tensor_names": codebook_tensor_names,
                        "tensor_payload_bytes": int(tensor_bytes),
                        "codebook_sidecar_bytes": int(codebook_bytes),
                        "total_bytes": int(tensor_bytes + codebook_bytes),
                    }
                layer_record[family] = records
            declaration_layers[str(layer)] = layer_record
        tensor_total = sum(
            int(group["tensor_payload_bytes"])
            for group in payload_groups.values()
        )
        codebook_total = sum(
            int(group["codebook_sidecar_bytes"])
            for group in payload_groups.values()
        )
        return (
            {
                "version": PER_EXPERT_FORMAT_GROUPS_VERSION,
                "layers": declaration_layers,
            },
            {
                "schema": "prismaquant.per_expert_format_group_payload.v1",
                "tensor_payload_bytes": int(tensor_total),
                "codebook_sidecar_bytes": int(codebook_total),
                "total_bytes": int(tensor_total + codebook_total),
                "groups": dict(sorted(payload_groups.items())),
            },
        )

    (
        _per_expert_wire_declaration,
        _per_expert_group_payload,
    ) = _per_expert_wire_declaration_and_payload()

    _declared_passthrough_units = _source_passthrough_units()
    # Floor block-FP8 units join the SAME declaration as allocated passthrough
    # units. They differ only in how they got here — the DP chose one and
    # skipped the other — and that distinction is invisible to, and irrelevant
    # for, the consumer: identical bytes, identical wire id, identical route.
    # Floor units face the SAME ship gate as allocated passthrough units, and
    # record the SAME acknowledgement. They reach it later only because
    # membership depends on what the planning pass consumed; the rule is
    # identical. Note the asymmetry with an allocated rung: shipping this one
    # undeclared is not a fallback, it is the bug — it drops the scale planes.
    # So the choice here is "declare and acknowledge" or "stop", never
    # "declare nothing and continue".
    if floor_fp8_units and _FP8_BLOCK_UE8M0_FORMAT in \
            ROUTE_PENDING_PASSTHROUGH_FORMATS:
        route_pending[_FP8_BLOCK_UE8M0_FORMAT] += len(floor_fp8_units)
        if not allow_route_pending_passthrough:
            raise ValueError(
                f"refusing to ship a route-pending source passthrough: "
                f"{_FP8_BLOCK_UE8M0_FORMAT} -> lane "
                f"{SOURCE_PASSTHROUGH_CONTRACTS[_FP8_BLOCK_UE8M0_FORMAT].serving_route} "
                f"({len(floor_fp8_units)} unallocated unit(s) the recipe "
                f"never assigned, e.g. "
                f"{sorted(floor_fp8_units)[:3]}). These are block-FP8 weights "
                f"with UE8M0 scale planes; they MUST be declared, because the "
                f"alternative is not 'ship them plainly' but 'ship them with "
                f"their scales dropped and silently cast to bf16'. Pass "
                f"--allow-route-pending-passthrough "
                f"(allow_route_pending_passthrough=True) to ship knowingly; "
                f"the acknowledgement is recorded in the artifact's "
                f"provenance.")
    for _floor_unit in floor_fp8_units:
        _previous = _declared_passthrough_units.get(_floor_unit)
        if _previous not in (None, _FP8_BLOCK_UE8M0_FORMAT):
            raise ValueError(
                f"{_floor_unit}: reached the verbatim floor as block-FP8 but "
                f"is already declared {_previous!r}; one unit has one "
                f"contract")
        _declared_passthrough_units[_floor_unit] = _FP8_BLOCK_UE8M0_FORMAT
    assert_routes_reconcile(
        **_route_reconciliation_sets(_declared_passthrough_units))

    post_allocation_refinement = None
    _meta_ref_stream = _recipe_payload.get("__prismaquant__", {})
    if isinstance(_meta_ref_stream, dict) and "post_allocation_refinement" in _meta_ref_stream:
        from prismaquant.cb_ldlq_refinement import validate_refinement_provenance

        post_allocation_refinement = validate_refinement_provenance(
            _meta_ref_stream.get("post_allocation_refinement"),
            where="export_nvfp4_cb_streaming post_allocation_refinement",
        )
    quant_config = build_quant_config(
        assignment=assignment,
        cb_targets=cb_targets,
        source_targets=source_targets,
        native_source_targets=native_source_targets,
        requant_targets=requant_targets,
        # Embedding units are packed like a stock target but claimed by the
        # `quantized_embedding` declaration, so they must not also appear in a
        # config group -- the consumer refuses a unit owned by two dispatches.
        stock_targets={q: f for q, f in stock_targets.items()
                       if q not in embedding_stock},
        quantized_embedding_units=embedding_stock or None,
        by_group=by_group,
        cb_group_target_names=cb_group_target_names,
        codebooks=codebooks,
        col_weights=col_weights,
        codebook_tensors_by_name=cb_tensor_blobs,
        ignore=ignore,
        codebook_file=codebook_file,
        scale_coding=scale_coding,
        codebook_source=serialization_context.codebook_source,
        serialized_payload_summary=serialized_payload_summary,
        serialization_context=serialization_context,
        cb_render_identity=_recipe_cb_render_identity,
        research_cost_selection=_research_cost_selection,
        post_allocation_refinement=post_allocation_refinement,
        activation_execution_contract=activation_execution_contract,
        git_commit=_git_commit(),
        cb_target_name=_cb_target_name,
        delegated_target_name=_delegated_target_name,
        source_target_name=_delegated_target_name,
        # The byte-verbatim lane keeps the CHECKPOINT spelling: those are the
        # names its tensors were actually written under, above.
        native_source_target_name=_base_name,
        # Same rule for the re-quant lane: its planes were written under
        # `_base_name(qname)`, so that is the name the group must claim.
        requant_target_name=_base_name,
        source_passthrough_units=_declared_passthrough_units,
        per_expert_format_groups=_per_expert_wire_declaration,
        route_pending_passthrough_acknowledged=sorted(route_pending),
        excluded_namespaces=excluded_namespaces,
        weight_only_stock_targets=sidecar_stock,
        streaming_provenance=True,
        # Release serving/performance evidence must be reconciled against the
        # finalized assignment, not a benchmark-authored route claim.  Keep
        # the compact per-Linear map in the inventory-bound quant_config so a
        # later validator can derive every serving-unit route independently.
        include_tensor_formats=True,
        tensor_formats=finalized_tensor_formats,
    )
    # Source DSpark is a metadata-only overlay. Quantized DSpark is a distinct
    # producer mode: physical tensor bases stay ``mtp.*`` while the CB groups
    # above were named in the construction namespace. Never apply the source
    # overlay to that artifact or it would double-declare every decoder unit.
    quant_config = apply_dspark_overlay_to_quant_config(
        quant_config, dspark_source_overlay
    )
    # The route census principle 12 requires next to any bpp or KL claim. It
    # rides the inventory-bound quant_config so it is part of the artifact's
    # identity rather than a log line, and its shape makes an unattested lane
    # impossible to read as a clean one: when the pinned release publishes no
    # eligibility table this payload carries units_unattested and NO
    # backed/fallback counters at all.
    quant_config.setdefault("provenance", {})["cb_route_status"] = (
        cb_route_status_provenance
    )
    if dspark_cb_sidecar:
        physical_cb_targets = sorted({_base_name(q) for q in cb_targets})
        expected_physical_cb_targets = list(
            dspark_cb_expected_physical_targets(source_config)
        )
        if physical_cb_targets != expected_physical_cb_targets:
            raise ValueError(
                "DSpark hybrid sidecar CB targets are not the exact 27-base "
                "contract: missing="
                f"{sorted(set(expected_physical_cb_targets) - set(physical_cb_targets))}, "
                "extra="
                f"{sorted(set(physical_cb_targets) - set(expected_physical_cb_targets))}"
            )
        construction_cb_targets = sorted({
            _cb_target_name(q) for q in cb_targets
        })
        for construction in construction_cb_targets:
            round_trip = dspark_cb_physical_output_for_construction_target(
                construction, source_config
            )
            if round_trip not in physical_cb_targets:
                raise AssertionError(
                    f"DSpark construction target {construction!r} round-trips "
                    f"to undeclared physical target {round_trip!r}"
                )
        bridge = build_dspark_target_bridge(
            source_config,
            contracted_cb_construction_targets=(
                [
                    dspark_cb_construction_target_for_physical_output(
                        str(physical), source_config
                    )
                    for physical in activation_execution_contract[
                        "target_names"
                    ]
                ]
                if activation_execution_contract is not None else ()
            ),
            activation_execution_contract=activation_execution_contract,
        )
        if bridge is not None:
            quant_config["dspark_target_bridge"] = bridge
        expected_source_units = sorted(dspark_hybrid_source_mapping.values())
        if sorted(_declared_passthrough_units) != expected_source_units:
            raise AssertionError(
                "DSpark hybrid source-passthrough declaration differs from "
                f"the exact construction set: observed="
                f"{sorted(_declared_passthrough_units)}, expected="
                f"{expected_source_units}"
            )
        quant_config["provenance"]["dspark_cb_sidecar"] = {
            "schema": "prismaquant.dspark_cb_sidecar.v1",
            "num_hidden_layers": (
                discovered_dspark_source_overlay.num_hidden_layers
            ),
            "n_mtp_layers": (
                discovered_dspark_source_overlay.n_mtp_layers
            ),
            "physical_namespace": "mtp.{stage}",
            "construction_namespace": (
                "model.layers.{num_hidden_layers+stage}"
            ),
            "physical_cb_targets": physical_cb_targets,
            "construction_cb_targets": construction_cb_targets,
            "source_passthrough_targets": sorted(
                dspark_hybrid_source_mapping
            ),
            "source_passthrough_physical_to_construction": dict(
                sorted(dspark_hybrid_source_mapping.items())
            ),
            "activation_bridge_present": bridge is not None,
        }
    _bind_source_model_identity_provenance(
        quant_config, source_model_identity
    )
    if _per_expert_group_payload is not None:
        quant_config["provenance"]["per_expert_format_group_payload"] = (
            _per_expert_group_payload
        )

    # DELTA-EXPORT: verify sampled copies + log the summary BEFORE writing (an
    # abort here leaves no partial artifact). No-op when reuse is disabled.
    if prior is not None:
        _reuse_verify_and_report(
            prior, reuse, reuse_verify, reuse_prior, col_weights,
            scale_coding, counts)

    print(f"[export-cb-stream] streaming {len(writer.names())} tensors ...",
          flush=True)
    from safetensors.torch import save_file

    def _assert_source_coverage_before_publish():
        expected_scope = {
            member for qname in cb_targets for member in _identity_scope(qname)
        }
        if (
            _recipe_cb_render_identity is not None
            and verified_cb_source_qnames != expected_scope
        ):
            raise AssertionError(
                "streaming CB source-value validation did not cover the exact "
                "assignment: missing="
                f"{sorted(expected_scope - verified_cb_source_qnames)[:8]}, "
                "extra="
                f"{sorted(verified_cb_source_qnames - expected_scope)[:8]}"
            )
        if ldlq_telemetry is not None:
            ldlq_telemetry.payload()

    writer.write(
        out_dir / SINGLE_CONTAINER_NAME,
        shard_bytes=int(shard_bytes),
        before_publish=_assert_source_coverage_before_publish,
    )
    published_containers = sorted(writer.last_weight_manifest_files)
    if cb_render_source_collector is not None:
        completed_render_identity = cb_render_source_collector.finalize()
        if not isinstance(_recipe_cb_render_identity, dict):
            raise AssertionError(
                "DSpark streaming render identity seed is not mutable"
            )
        # ``build_quant_config`` intentionally retains the identity object.
        # Complete it before quant_config/shipcard publication so the artifact
        # carries the exact decoded source digests observed during this one
        # render, without retaining multi-billion-parameter tensors in RAM.
        _recipe_cb_render_identity.clear()
        _recipe_cb_render_identity.update(completed_render_identity)
        quant_config["provenance"]["render_identity_verified"] = True
        quant_config["provenance"]["cb_render_identity"] = (
            _recipe_cb_render_identity
        )
        quant_config["provenance"]["dspark_render_attestation"] = {
            "schema": _DSPARK_RENDER_RECIPE_SCHEMA,
            "source_binding": _DSPARK_RENDER_SOURCE_BINDING,
            # Persist the complete, already validated recipe rather than only
            # its digest.  The digest is useful for compact receipts, but a
            # production consumer must be able to replay the source/config/
            # assignment/imatrix bindings without relying on an external
            # layer-config file that may have moved after export.
            "recipe": dict(_dspark_render_recipe),
            "recipe_sha256": _canonical_json_digest(_dspark_render_recipe),
            "source_weights_sha256": _recipe_cb_render_identity[
                "source_weights_sha256"
            ],
            "source_weights_entries": len(
                _recipe_cb_render_identity["source_weights_content_sha256"]
            ),
        }
    if not writer.last_weight_manifest_files:
        raise AssertionError(
            "streaming writer published without an exact content digest"
        )
    # Layout-invariant payload identity, hashed in the same pass that hashed
    # the containers.  See `shard_layout.tensor_payload_identity` for why the
    # artifact needs an identity `model_sha` cannot provide.
    quant_config["provenance"]["tensor_payload_identity"] = (
        tensor_payload_identity(writer.last_tensor_content_sha256)
    )
    if warm_session is not None:
        warm_provenance = warm_session.provenance()
        quant_config["provenance"]["encoder_warm_start"] = warm_provenance
        counts.update(warm_provenance)
        print(
            f"[export-cb-stream] encoder warm state: {warm_provenance}",
            flush=True,
        )
    if cb_tensor_blobs:
        save_file(cb_tensor_blobs, str(out_dir / codebook_file),
                  metadata={"format": "pt"})
    src_config = model_dir / "config.json"
    config = json.loads(src_config.read_text()) if src_config.exists() else {}
    config["quantization_config"] = {
        "quant_method": "gridbook", "format": "nvfp4_cb",
        "config_file": "quant_config.json"}
    config = apply_dspark_overlay_to_model_config(
        config,
        (
            discovered_dspark_source_overlay
            if dspark_cb_sidecar else dspark_source_overlay
        ),
    )
    if dspark_body_only:
        # A source artifact may already carry a stamp from a prior full DSpark
        # export.  Subset/exclude is allowed to produce a body-only artifact,
        # but that artifact must not promise draft modules whose bytes were
        # deliberately omitted.
        config.pop("n_mtp_layers", None)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    for aux in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                "special_tokens_map.json", "generation_config.json",
                "vocab.json", "merges.txt", "chat_template.jinja",
                "chat_template.json", "preprocessor_config.json",
                "video_preprocessor_config.json", "processor_config.json"):
        p = model_dir / aux
        if p.exists():
            (out_dir / aux).write_bytes(p.read_bytes())
    if ldlq_telemetry is not None:
        ldlq_telemetry.publish(out_dir, quant_config)
    # Open the refusal record before inventory finalization: the preliminary
    # quant_config binds the CB identity, while shipcard.json itself must be
    # measured by the recursive inventory and the hard artifact budget.
    from prismaquant.shipcard import (
        WEIGHT_CONTENT_MANIFEST_SCHEMA,
        open_cb_export_shipcard,
    )

    routed_book_keyings = sorted(
        ({ROUTED_BOOK_KEYING_STACK} if pooled_stack_cells else set())
        | ({ROUTED_BOOK_KEYING_ROLE} if expert_role_plans else set())
    )
    open_cb_export_shipcard(
        out_dir,
        quant_config,
        source_model=model_dir,
        layer_config_path=layer_config_path,
        exporter="export_nvfp4_cb_streaming",
        build_extra={
            # Campaign rule R1, on the card next to bpp: how many fused routed
            # weights name one book, how many name several, and whether an
            # operator knowingly shipped the split ones.
            "routed_codebook_books": {
                "keying": routed_book_keyings,
                "pooled_stack_units": len(pooled_stack_cells),
                "per_role_units": len(expert_role_plans),
                "fused_targets_with_split_books": sorted(split_book_targets),
                "per_role_books_override": bool(
                    split_book_targets and allow_per_role_books
                ),
            },
        },
        weight_content_manifest={
            "schema": WEIGHT_CONTENT_MANIFEST_SCHEMA,
            "algorithm": "sha256",
            "files": dict(sorted(writer.last_weight_manifest_files.items())),
        },
    )
    # Persist and assert a final filesystem inventory distinct from the CB
    # tensor-data payload contract.  This includes both safetensors headers,
    # JSON configs, tokenizer assets, and all other regular output files.
    finalize_cb_export_artifact_inventory(
        out_dir,
        quant_config,
        serialized_payload=serialized_payload_summary,
        cb_tensor_names=sorted(cb_output_tensor_names),
        codebook_file=codebook_file,
        expected_model_files=published_containers,
        whole_artifact_budget_bytes=(
            int(_whole_artifact_budget["budget_bytes"])
            if _whole_artifact_budget is not None
            else None
        ),
    )
    # FINAL GATE, read back off the PUBLISHED artifact rather than from the
    # planning state that produced it. Everything above knows what it meant to
    # write; this asks the only question a consumer can ask — for every
    # scale-bearing weight actually on disk, is there a mechanism that decodes
    # it? Headers only, so it costs a few opens even on a 92 GB artifact.
    from prismaquant.artifact_completeness import assert_artifact_complete

    _verbatim = getattr(profile, "source_passthrough_prefixes", None)
    verbatim_prefixes = (
        () if (dspark_source_overlay is not None or dspark_cb_sidecar) else
        (tuple(_verbatim()) if callable(_verbatim) else ("mtp.",))
    )
    completeness = assert_artifact_complete(
        out_dir, verbatim_prefixes=verbatim_prefixes)
    if excluded_namespaces:
        print(f"[export-cb-stream] excluded namespace(s) "
              f"{list(excluded_namespaces)}: {counts['excluded']} tensor(s) "
              f"omitted from the artifact (recorded in provenance)",
              flush=True)
    print(f"[export-cb-stream] completeness: "
          f"{len(completeness.passthrough_units)} declared passthrough, "
          f"{len(completeness.verbatim_namespace_units)} verbatim-namespace "
          f"unit(s); no orphan scale planes", flush=True)
    return dict(counts)


def _stream_pack_target(skeleton, profile, resolved, qname, grid, mode, k,
                        codebook, cw, scale_sweep, coding, shape, device,
                        encode_tier, cb_render_identity,
                        cb_render_source_collector,
                        verified_source_qnames, member_qnames=None, *,
                        expert_roles=None, learned_bundle=None,
                        all_col_weights=None,
                        warm_session=None, format_name=None,
                        ldlq_activation_loader=None,
                        ldlq_telemetry=None):
    """Pack ONE target, streaming experts. Returns (packed uint8 (rows,bytes)
    or (E,out,bytes), scale-plane fp32 or None). Per-expert scales make
    per-expert packing byte-identical to whole-stack packing."""
    kind, h = resolved
    cbook = _to_device(codebook, device)
    if kind == "tensor":
        # Keep the decoded host value intact until the source-value contract
        # has observed it.  ``_encode_prefetched_cb_tensor`` owns the H2D; if
        # we move first, its canonical FP32 digest immediately copies the
        # entire tensor back to host on a CUDA export.
        w = skeleton.dequant_weight(h)
        return _encode_prefetched_cb_tensor(
            w,
            qname=qname,
            grid=grid,
            mode=mode,
            k=k,
            codebook=cbook,
            cw=cw,
            scale_sweep=scale_sweep,
            coding=coding,
            device=device,
            encode_tier=encode_tier,
            cb_render_identity=cb_render_identity,
            cb_render_source_collector=cb_render_source_collector,
            verified_source_qnames=verified_source_qnames,
            warm_session=warm_session,
            format_name=format_name,
            ldlq_activation_loader=ldlq_activation_loader,
            ldlq_telemetry=ldlq_telemetry,
        )
    # Experts: build ONE layer's stack (fp4 derives a single per-tensor global
    # over the whole stack, so per-expert packing would diverge — the stack is
    # the byte-identity working set) and pack it whole, exactly as the
    # in-memory exporter packs a pre-stacked 3-D tensor. Peak = one MoE layer's
    # experts, not the model.
    prefix, packed_proj, grp = h
    projections = _packed_expert_projection_names(profile, packed_proj)
    expert_ids = (
        sorted({expert_id for _projection, expert_id in member_qnames})
        if member_qnames is not None
        else list(range(_n_experts(grp, projections)))
    )
    n = len(expert_ids)
    on_member = None
    if (
        (cb_render_identity is not None or cb_render_source_collector is not None)
        and member_qnames is not None
    ):
        # A per-expert checkpoint's render identity is keyed PER EXPERT (the
        # cost rows and the imatrix are too), so the stack is verified member
        # by member as it is decoded. Hashing the concatenated stack instead
        # would certify a name the recipe never priced.
        from prismaquant.production_weight_cache import (
            validate_cb_render_source_weight,
        )

        def on_member(proj, e, _base, decoded, _q=member_qnames):
            member = _q[(proj, e)]
            if cb_render_source_collector is not None:
                cb_render_source_collector.observe(member, decoded)
            else:
                validate_cb_render_source_weight(
                    cb_render_identity,
                    member,
                    decoded,
                    where="export_nvfp4_cb_streaming expert stack member",
                )
            verified_source_qnames.add(member)

    # Fill a PREALLOCATED device buffer one expert at a time. Building a list
    # of E decoded experts, stacking it, then copying the stack to the device
    # holds three copies of the whole stack at once — on DSv4-Flash that is
    # 3 x 17.2 GB for a single `gate_up` group (256 x 4096 x 4096 fp32) and the
    # box has ~23 GB free while the cost run holds the rest. One buffer plus
    # one resident expert is the actual working set the module docstring
    # claims.
    first_expert = expert_ids[0]
    first = _expert_weight(skeleton, profile, prefix, packed_proj, grp,
                           first_expert,
                           on_member=on_member)
    w = torch.empty((n, *first.shape), dtype=first.dtype, device=device)
    w[0] = first.to(device)
    del first
    for local_index, expert_id in enumerate(expert_ids[1:], start=1):
        chunk = _expert_weight(skeleton, profile, prefix, packed_proj, grp,
                               expert_id,
                               on_member=on_member)
        w[local_index] = chunk.to(device)
        del chunk
    if (
        (cb_render_identity is not None or cb_render_source_collector is not None)
        and member_qnames is None
    ):
        from prismaquant.production_weight_cache import (
            validate_cb_render_source_weight,
        )
        if cb_render_source_collector is not None:
            cb_render_source_collector.observe(qname, w)
        else:
            validate_cb_render_source_weight(
                cb_render_identity,
                qname,
                w,
                where="export_nvfp4_cb_streaming expert stack",
            )
        verified_source_qnames.add(qname)
    if expert_roles is not None:
        return _pack_routed_moe_role_books(
            w,
            qname=qname,
            roles=tuple(expert_roles),
            grid=grid,
            mode=mode,
            k=k,
            scale_sweep=scale_sweep,
            coding=coding,
            device=device,
            encode_tier=encode_tier,
            learned_bundle=learned_bundle,
            all_col_weights=all_col_weights,
            ldlq_activation_loader=ldlq_activation_loader,
            ldlq_telemetry=ldlq_telemetry,
        )
    packed, fields = _pack_with_optional_warm_state(
        w, qname=qname, format_name=format_name, grid=grid, mode=mode, k=k,
        col_weights=cw.to(device), codebook=cbook,
        scale_sweep=scale_sweep, scale_coding=coding,
        encode_tier=encode_tier, warm_session=warm_session,
        ldlq_activation_rows=(
            ldlq_activation_loader.load(qname, stack_size=int(w.shape[0]))
            if ldlq_activation_loader is not None else None
        ),
        ldlq_telemetry=ldlq_telemetry,
    )
    packed = packed.reshape(w.shape[0], w.shape[1], -1)
    scale = (fields["scales"].reshape(*w.shape[:-1]).cpu()
             if grid == "fp8" else None)
    return packed.to(torch.uint8).cpu().contiguous(), scale


def _pack_routed_moe_role_books(
    weight,
    *,
    qname,
    roles,
    grid,
    mode,
    k,
    scale_sweep,
    coding,
    device,
    encode_tier,
    learned_bundle,
    all_col_weights,
    ldlq_activation_loader=None,
    ldlq_telemetry=None,
):
    """Encode contiguous expert roles with their exact immutable books."""

    if (
        grid != "fp8"
        or mode != "product"
        or int(k) not in ROUTED_MOE_CBL_BANK_RUNGS
    ):
        raise ValueError(
            f"{qname}: per-role expert books require the banked "
            "FP8-CB K28--K33 range"
        )
    if learned_bundle is None or all_col_weights is None:
        raise AssertionError(f"{qname}: per-role expert plan lost bundle inputs")
    role_slices = split_role_rows(weight, tuple(roles))
    activation_rows = (
        ldlq_activation_loader.load(qname, stack_size=int(weight.shape[0]))
        if ldlq_activation_loader is not None else None
    )
    packed_parts = []
    scale_parts = []
    for role, role_weight in role_slices:
        if role.format_name.rsplit("K", 1)[-1] != str(int(k)):
            raise ValueError(
                f"{qname}: role {role.qname} format {role.format_name} "
                f"disagrees with physical K{k}"
            )
        if len(role.member_qnames) != int(role_weight.shape[0]):
            raise ValueError(
                f"{role.qname}: member count does not match expert population"
            )
        for expert_index, member in enumerate(role.member_qnames):
            if member not in all_col_weights:
                raise ValueError(
                    f"{role.qname}: learned role member {member!r} has no "
                    "col_weights entry"
                )
            learned_bundle.validate_inputs(
                member,
                weight=role_weight[expert_index],
                col_weights=all_col_weights[member],
            )
        packed, fields = _pack_with_optional_warm_state(
            role_weight,
            qname=role.qname,
            format_name=role.format_name,
            grid=grid,
            mode=mode,
            k=k,
            col_weights=role.col_weights.to(device),
            codebook=_to_device(role.codebook, device),
            scale_sweep=scale_sweep,
            scale_coding=coding,
            encode_tier=encode_tier,
            # Collapsed stacks already cold-fallback in the existing warm-state
            # contract. A logical role has no independently measured argmin.
            warm_session=None,
            ldlq_activation_rows=activation_rows,
            ldlq_telemetry=ldlq_telemetry,
        )
        packed_parts.append(packed.reshape(
            role_weight.shape[0], role_weight.shape[1], -1
        ))
        scale_parts.append(fields["scales"].reshape(
            role_weight.shape[0], role_weight.shape[1]
        ))
    return (
        torch.cat(packed_parts, dim=1).to(torch.uint8).cpu().contiguous(),
        torch.cat(scale_parts, dim=1).to(torch.float32).cpu().contiguous(),
    )


def _prefetch_source_weight(skeleton, weight_key: str, device) -> torch.Tensor:
    """Read one dense source tensor ahead, pinning host storage for CUDA H2D."""

    return _pin_prefetched_tensor(skeleton.dequant_weight(weight_key), device)


def _pin_prefetched_tensor(weight: torch.Tensor, device) -> torch.Tensor:
    """Pin a reader result only when the encode target is CUDA-backed."""

    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available() \
            and weight.device.type == "cpu" and not weight.is_pinned():
        weight = weight.pin_memory()
    return weight


def _encode_prefetched_cb_tensor(
    weight: torch.Tensor,
    *,
    qname,
    grid,
    mode,
    k,
    codebook,
    cw,
    scale_sweep,
    coding,
    device,
    encode_tier,
    cb_render_identity,
    cb_render_source_collector,
    verified_source_qnames,
    warm_session=None,
    format_name=None,
    ldlq_activation_loader=None,
    ldlq_telemetry=None,
):
    """Encode the same dense tensor math from a reader-prefetched host value."""

    # The source identity is defined over the decoded producer value.  Reader
    # prefetch supplies that value on host, so observe/validate it before H2D.
    # Besides avoiding a full D2H round trip, this makes the one-pass evidence
    # independent of encode-device allocator state.
    if cb_render_identity is not None or cb_render_source_collector is not None:
        from prismaquant.production_weight_cache import (
            validate_cb_render_source_weight,
        )
        if cb_render_source_collector is not None:
            cb_render_source_collector.observe(qname, weight)
        else:
            validate_cb_render_source_weight(
                cb_render_identity,
                qname,
                weight,
                where="export_nvfp4_cb_streaming source tensor",
            )
        verified_source_qnames.add(qname)
    w = weight.to(
        device,
        non_blocking=bool(weight.device.type == "cpu" and weight.is_pinned()),
    )
    cbook = _to_device(codebook, device)
    packed, fields = _pack_with_optional_warm_state(
        w,
        qname=qname,
        format_name=format_name,
        grid=grid,
        mode=mode,
        k=k,
        col_weights=cw.to(device),
        codebook=cbook,
        scale_sweep=scale_sweep,
        scale_coding=coding,
        encode_tier=encode_tier,
        warm_session=warm_session,
        ldlq_activation_rows=(
            ldlq_activation_loader.load(
                qname,
                stack_size=(int(w.shape[0]) if w.dim() == 3 else None),
            ) if ldlq_activation_loader is not None else None
        ),
        ldlq_telemetry=ldlq_telemetry,
    )
    if w.dim() == 3:
        packed = packed.reshape(w.shape[0], w.shape[1], -1)
    scale = (
        fields["scales"].reshape(*w.shape[:-1]).cpu()
        if grid == "fp8"
        else None
    )
    return packed.to(torch.uint8).cpu().contiguous(), scale


def _pack_with_optional_warm_state(
    weight, *, qname, format_name, grid, mode, k, col_weights, codebook,
    scale_sweep, scale_coding, encode_tier, warm_session,
    ldlq_activation_rows=None,
    ldlq_telemetry=None,
):
    """Run normal assignment/packing, optionally seeded by a stored argmin."""
    from prismaquant.cb_warm_state import (
        CBEncodedPayload,
        selected_scale_state,
    )

    def encode(warm_scale_state=None):
        gate_info: dict[str, object] | None = (
            {} if ldlq_activation_rows is not None else None
        )
        packed, fields = cb.nvfp4_cb_pack(
            weight,
            k,
            grid=grid,
            mode=mode,
            col_weights=col_weights,
            codebook=codebook,
            scale_sweep=scale_sweep,
            scale_coding=scale_coding,
            encode_tier=encode_tier,
            warm_scale_state=warm_scale_state,
            ldlq=ldlq_activation_rows is not None,
            activation_rows=ldlq_activation_rows,
            ldlq_gate_info_out=gate_info,
        )
        rendered = {"packed": packed}
        if grid == "fp8":
            rendered["weight_scale"] = fields["scales"]
        return CBEncodedPayload(
            value=(packed, fields, gate_info),
            selected_scale=selected_scale_state(fields),
            rendered=rendered,
        )

    if warm_session is None:
        selected = encode().value
    else:
        payload = warm_session.encode(
            qname,
            format_name,
            full_encode=lambda: encode(),
            seeded_encode=lambda state: encode(state),
        )
        selected = payload.value
    packed, fields, gate_info = selected
    if ldlq_telemetry is not None:
        if gate_info is None:
            raise AssertionError(f"{qname}: LDLQ telemetry has no gate result")
        ldlq_telemetry.record(
            qname=qname,
            shape=tuple(int(dim) for dim in weight.shape),
            grid=grid,
            mode=mode,
            k=k,
            gate_info=gate_info,
        )
    return packed, fields


def _train_shared_codebook_streaming(skeleton, profile, expert_groups,
                                     resolve_target, qnames, col_weights, *,
                                     grid, mode, k, seed, iters, train_cap,
                                     device, members_by_target=None):
    """Bounded-pool learned codebook: sample scaled vectors from each target's
    source (streamed) up to ``train_cap`` total, then train — never all
    role weights resident. For a small role (< train_cap) the pooled set
    equals the in-memory exporter's, so the codebook is identical."""
    from prismaquant.export_nvfp4_cb import _train_shared_codebook
    weights, cws = [], []
    for q in qnames:
        kind, h = resolve_target(q)
        if kind == "tensor":
            weights.append(skeleton.dequant_weight(h).to(device))
        else:
            prefix, packed_proj, grp = h
            member_qnames = (members_by_target or {}).get(q)
            expert_ids = (
                sorted({expert_id for _projection, expert_id in member_qnames})
                if member_qnames is not None else None
            )
            weights.append(_stacked_source_weight(
                skeleton, profile, prefix, packed_proj, grp,
                expert_ids=expert_ids).to(device))
        cws.append(col_weights[q].to(device))
    return _train_shared_codebook(
        weights, cws, grid=grid, mode=mode, k=k, seed=seed, iters=iters,
        train_cap=train_cap)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Streaming NVFP4-CB exporter")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer-config", required=True)
    ap.add_argument(
        "--per-expert-config",
        default=None,
        help="PROPOSED v1 split-stack mode: JSON qname->format allocation; "
        "routed expert rows override --layer-config and emit one sub-stack "
        "per (layer, w13/w2 family, format)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES,
                    help="Approx per-shard size in bytes (default 1 GiB), the "
                         "same flag, default, and partition rule as "
                         "export_native_compressed. A single tensor larger "
                         "than this still gets its own shard. One resulting "
                         "shard is published as model.safetensors with no "
                         "index; pass a value at least as large as the "
                         "artifact to reproduce the legacy single-container "
                         "layout.")
    ap.add_argument("--col-weights", required=True,
                    help="pickle {qname: per-column importance}")
    ap.add_argument(
        "--activation-cache-dir",
        default=None,
        help="probe activation cache used to calibrate the versioned static "
        "W4A4 input_global_scale contract",
    )
    ap.add_argument(
        "--activation-scale-policy",
        default=None,
        choices=sorted((
            "legacy_6_over_calibration_amax.v1",
            "full_e4m3_range_448x6_over_calibration_amax.v1",
            "mse_grid_calibrated.v1",
        )),
    )
    ap.add_argument("--codebook-source", default="lattice",
                    choices=["lattice", "learned"])
    ap.add_argument("--codebook-iters", type=int, default=4)
    ap.add_argument("--codebook-seed", type=int, default=0)
    ap.add_argument("--no-scale-sweep", action="store_true")
    ap.add_argument(
        "--allow-unstamped-research",
        action="store_true",
        help="unsafe research-only escape hatch for a bare CB assignment; "
        "production recipes must carry a source/imatrix-complete render "
        "identity",
    )
    ap.add_argument(
        "--allow-research-cost-selection",
        action="store_true",
        help="explicitly acknowledge export of an allocation derived from "
             "the sanctioned study-grade assembled cost table; recorded in "
             "artifact provenance",
    )
    ap.add_argument(
        "--allow-unbacked-route",
        default=None,
        metavar="REASON",
        help="ship even though a selected unit has NO backed serving route "
        "under the pinned Gridbook release (campaign rule R3, principle 9). "
        "Takes the REASON, not a flag: it is stamped into the artifact's "
        "provenance and read by whoever inherits the serving problem. May also "
        f"be set with {ROUTE_OVERRIDE_ENV}=<reason>.",
    )
    ap.add_argument(
        "--non-native-target",
        default=None,
        metavar="PLATFORM",
        help="declare that this artifact targets a platform whose native lane "
        "does not exist, so unbacked routes are expected rather than a defect. "
        "Stamped on the artifact; a win on a non-native kernel is not a win on "
        f"the named hardware (principle 12). Env: {NON_NATIVE_TARGET_ENV}.",
    )
    ap.add_argument(
        "--allow-route-pending-passthrough",
        action="store_true",
        default=_route_pending_ack_from_env(),
        help="ship a source-passthrough rung whose serve route is not yet "
        "validated (allocator_candidates.ROUTE_PENDING_PASSTHROUGH_FORMATS). "
        "Refused by default; the acknowledgement is recorded in the artifact "
        f"provenance. May also be set with {_ROUTE_PENDING_ACK_ENV}=1 so a "
        "driver script can pass it explicitly; the env form announces itself "
        "on stderr and is OFF unless set to exactly '1'.",
    )
    ap.add_argument(
        "--allow-per-role-books",
        action="store_true",
        help="ship fused routed weights whose scheme names more than one "
        "codebook (books burned per (layer, projection, rung) rather than "
        "pooled per (layer, stack, rung), campaign rule R1). Refused by "
        "default; passing it stamps the acknowledgement onto the shipcard.",
    )
    ap.add_argument(
        "--exclude-namespace",
        action="append",
        default=None,
        dest="exclude_namespaces",
        help="OMIT every tensor whose checkpoint name starts with this prefix "
        "from the artifact entirely — no tensor, no index entry, no `ignore` "
        "line, no declaration. Repeatable. Legal only for floor/verbatim "
        "tensors: a prefix matching any unit the layer_config allocates is a "
        f"hard refusal. May also be set with {_EXCLUDE_NAMESPACES_ENV} as a "
        "comma-separated list; empty/unset excludes nothing.",
    )
    ap.add_argument(
        "--scale-coding",
        default=cb.SCALE_CODING_TWO_TIER,
        choices=[cb.SCALE_CODING_V1, cb.SCALE_CODING_TWO_TIER],
        help="production layout-v2 two-tier coding (default), or explicit "
        "legacy v1 for backward-compatible artifacts",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--subset-prefix", action="append", default=None,
                    metavar="PREFIX",
                    help="opt-in: export ONLY tensors under this checkpoint "
                         "prefix (repeatable), e.g. 'model.layers.80.' for the "
                         "MTP sidecar; every allocation target must fall within "
                         "it. Default: whole-model passthrough.")
    ap.add_argument(
        "--dspark-cb-sidecar",
        action="store_true",
        help="emit a separate, quantized DeepSeek-V4 DSpark draft sidecar; "
        "requires exactly --subset-prefix mtp. and the validated released "
        "three-stage source topology",
    )
    ap.add_argument("--reuse-prior", default=None, metavar="DIR",
                    help="reserved DELTA-EXPORT input; currently fails closed "
                         "until exact source/imatrix/codebook/exporter identity "
                         "is implemented. Env PRISMAQUANT_EXPORT_REUSE_PRIOR "
                         "is also rejected.")
    ap.add_argument("--reuse-verify", type=int, default=None, metavar="N",
                    help="reserved compatibility option while DELTA-EXPORT is "
                         "disabled (default 3; env "
                         "PRISMAQUANT_EXPORT_REUSE_VERIFY)")
    ap.add_argument(
        "--warm-state-dir",
        default=os.environ.get("PRISMAQUANT_CB_WARM_STATE_DIR") or None,
        help="optional CB encoder scale-search sidecar directory; defaults "
        "to PRISMAQUANT_CB_WARM_STATE_DIR when set",
    )
    ap.add_argument(
        "--warm-verify-sample",
        type=int,
        default=32,
        metavar="N",
        help="full-sweep and byte-verify N randomly sampled warm units "
        "(default: 32; any mismatch aborts)",
    )
    args = ap.parse_args(argv)
    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("export_nvfp4_cb_streaming")
    reuse_prior = args.reuse_prior or os.environ.get(
        "PRISMAQUANT_EXPORT_REUSE_PRIOR") or None
    reuse_verify = (args.reuse_verify if args.reuse_verify is not None
                    else int(os.environ.get(
                        "PRISMAQUANT_EXPORT_REUSE_VERIFY", "3")))
    if torch.cuda.is_available():
        # Box-safety net on the unified pool: a runaway allocation must raise
        # a clean torch OOM (with the offending tensor in the traceback), not
        # drive the whole box to a kernel global OOM (3x on 2026-07-19).
        torch.cuda.set_per_process_memory_fraction(0.75)
    with open(args.col_weights, "rb") as fh:
        col_weights = {k: torch.as_tensor(v) for k, v in pickle.load(fh).items()}
    spec = {"source": args.codebook_source}
    if args.codebook_source == "learned":
        spec.update(train=True, iters=args.codebook_iters,
                    seed=args.codebook_seed)
    counts = export_nvfp4_cb_streaming(
        args.model_dir, args.layer_config, args.out, col_weights,
        shared_codebook_spec=spec, device=args.device,
        scale_sweep=not args.no_scale_sweep, scale_coding=args.scale_coding,
        subset_prefixes=args.subset_prefix, reuse_prior=reuse_prior,
        reuse_verify=reuse_verify,
        allow_unstamped_research=args.allow_unstamped_research,
        allow_research_cost_selection=args.allow_research_cost_selection,
        allow_route_pending_passthrough=args.allow_route_pending_passthrough,
        allow_per_role_books=args.allow_per_role_books,
        allow_unbacked_route=args.allow_unbacked_route,
        non_native_target=args.non_native_target,
        exclude_namespaces=args.exclude_namespaces,
        activation_cache_dir=args.activation_cache_dir,
        activation_scale_policy=args.activation_scale_policy,
        per_expert_config_path=args.per_expert_config,
        warm_state_dir=args.warm_state_dir,
        warm_verify_sample=args.warm_verify_sample,
        dspark_cb_sidecar=args.dspark_cb_sidecar,
        shard_bytes=args.shard_bytes)
    size = sum(p.stat().st_size for p in Path(args.out).glob("*")) / 1e9
    print(f"wrote {args.out} ({size:.3f} GB)")
    for fmt, n in sorted(counts.items()):
        print(f"  {fmt}: {n}")


if __name__ == "__main__":
    main()
