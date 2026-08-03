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
import hashlib
import inspect
import json
import os
import pickle
import re
import struct
from collections import Counter
from functools import wraps
from pathlib import Path

import torch
from safetensors import safe_open

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.allocator_candidates import (
    ROUTE_PENDING_PASSTHROUGH_FORMATS,
    SOURCE_PASSTHROUGH_CONTRACTS,
)
from prismaquant.cb_export_config import (
    SOURCE_PASSTHROUGH_EXPORT_FORMATS,
    STREAMING_REQUANT_EXPORT_FORMATS,
    parse_source_passthrough_declaration,
    build_cb_scheme,
    build_quant_config,
    cb_scheme_reuse_signature,
    codebook_tensor_names as _codebook_tensor_names,
    codebook_tensors as _codebook_tensors,
    source_passthrough_wire,
)
from prismaquant.format_registry import (
    canonical_format_name,
    get_format as _fr_get_format,
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
from prismaquant.layer_config import load_assignment
from prismaquant.model_profiles import detect_profile
from prismaquant.export_output_safety import (
    prepare_fresh_export_directory,
    transactional_directory_output,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    cb_payload_summary,
    cb_serialization_metadata_from_assignment_payload,
    cb_tensor_payload_breakdown,
    finalize_cb_export_artifact_inventory,
    resolve_cb_encode_tier,
    whole_artifact_budget_from_assignment_payload,
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


class _StreamWriter:
    """Two-pass safetensors writer. ``add`` records (name, dtype, shape) and a
    zero-arg ``producer`` that yields the tensor at write time; ``write`` lays
    out contiguous offsets, writes the header, then streams every producer's
    bytes in order — one output tensor resident at a time."""

    def __init__(self):
        self._entries: list[tuple[str, torch.dtype, tuple, object, object]] = []

    def add(self, name, dtype, shape, producer, copy_src=None):
        """Record an output tensor. ``producer`` yields it at write time; when
        ``copy_src=(path, file_offset, nbytes)`` is given (DELTA-EXPORT reuse)
        those raw bytes are streamed straight from a prior artifact's shard file
        instead — ``producer`` is then unused (may be None)."""
        self._entries.append((name, dtype, tuple(int(d) for d in shape),
                              producer, copy_src))

    def names(self) -> list[str]:
        return [e[0] for e in self._entries]

    def write(self, path: Path, *, before_publish=None) -> None:
        header: dict[str, dict] = {}
        off = 0
        for name, dtype, shape, _, _ in self._entries:
            nb = _nbytes(dtype, shape)
            if name in header:
                # The header is a dict but the data stream is not: a duplicate
                # name keeps only the LAST span while both blobs are still
                # written, producing a file whose offsets are silently wrong
                # from that point on. Cheap to check, and the passthrough lane
                # adds tens of thousands of names from a second namespace.
                raise AssertionError(
                    f"{name}: planned twice; two emit paths claim the same "
                    "output tensor")
            header[name] = {"dtype": _ST_DTYPE[dtype], "shape": list(shape),
                            "data_offsets": [off, off + nb]}
            off += nb
        header["__metadata__"] = {"format": "pt", "quant_method": "gridbook"}
        hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
        data0 = 8 + len(hjson)

        # A safetensors header binds names/dtypes/shapes, not the source
        # weights, imatrix, codebooks, or exporter implementation.  Reusing a
        # same-shaped partial file can therefore preserve bytes produced by a
        # different render while every final span/size assertion still passes.
        # Resume stays disabled until the header carries one immutable digest
        # covering all of those producer inputs.
        if os.path.lexists(path):
            raise RuntimeError(
                f"{path}: refusing an unbound streaming resume. The existing "
                "file header does not prove source/imatrix/codebook/exporter "
                "identity; use a fresh output directory."
            )
        temp_path = path.with_name(f".{path.name}.tmp")
        if os.path.lexists(temp_path):
            raise RuntimeError(
                f"{temp_path}: refusing to overwrite a stale or aliased "
                "streaming-export temporary file"
            )

        cuda = torch.cuda.is_available()
        owns_temp = False
        try:
            with open(temp_path, "xb") as f:
                owns_temp = True
                f.write(struct.pack("<Q", len(hjson)))
                f.write(hjson)
                for i, (name, dtype, shape, producer, copy_src) in enumerate(
                        self._entries):
                    if copy_src is not None:
                        # DELTA-EXPORT: stream raw bytes from a prior artifact.
                        src_path, foff, nb = copy_src
                        if nb != _nbytes(dtype, shape):
                            raise AssertionError(
                                f"{name}: copy_src {nb}B != declared "
                                f"{_nbytes(dtype, shape)}B")
                        with open(src_path, "rb") as sf:
                            sf.seek(foff)
                            remaining = nb
                            while remaining:
                                chunk = sf.read(min(remaining, 1 << 24))
                                if not chunk:
                                    raise AssertionError(
                                        f"{name}: prior artifact truncated at "
                                        f"offset {foff} (needed {nb}B)")
                                f.write(chunk)
                                remaining -= len(chunk)
                        if i % 50 == 0 or nb > (1 << 30):
                            print(f"[export-cb-stream] {i + 1}/"
                                  f"{len(self._entries)} {name} copied "
                                  f"{nb / 2**30:.2f}G from prior", flush=True)
                        continue
                    t = producer()
                    if t.dtype != dtype or tuple(t.shape) != shape:
                        raise AssertionError(
                            f"{name}: produced {t.dtype}{tuple(t.shape)} != "
                            f"declared {dtype}{shape}")
                    b = _raw_bytes(t)
                    if len(b) != _nbytes(dtype, shape):
                        raise AssertionError(f"{name}: byte count mismatch")
                    f.write(b)
                    del t, b
                    if cuda:
                        # Unified-memory hygiene: differently-shaped 10GB-class
                        # pack transients must not accumulate as cached segments.
                        torch.cuda.empty_cache()
                        if i % 20 == 0 or _nbytes(dtype, shape) > (1 << 30):
                            print(f"[export-cb-stream] {i + 1}/"
                                  f"{len(self._entries)} {name} "
                                  f"cuda alloc "
                                  f"{torch.cuda.memory_allocated() / 2**30:.1f}G "
                                  f"reserved "
                                  f"{torch.cuda.memory_reserved() / 2**30:.1f}G",
                                  flush=True)
            if before_publish is not None:
                before_publish()
            os.replace(temp_path, path)
            owns_temp = False
        except Exception:
            if owns_temp and os.path.lexists(temp_path):
                temp_path.unlink()
            raise


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


def _member_serialized_shapes(packed_qname, member_qnames, expert_groups,
                              skeleton, profile):
    """``{member recipe qname: decoded 2-D shape}`` for one collapsed stack.

    Read from the checkpoint rather than divided out of the stack shape, so a
    fused parent whose projections are NOT equal-width is described exactly."""
    if not member_qnames:
        return {}
    prefix, packed_proj = packed_qname.rsplit(".", 1)
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
        packed_proj = packed_qname.rsplit(".", 1)[1]
        projections = _packed_expert_projection_names(profile, packed_proj)
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


def _stacked_source_weight(skeleton, profile, prefix, packed_proj, members) -> \
        torch.Tensor:
    """Materialise the full stacked source weight (E, out, in) for a packed
    expert group — used only where a stack must be resident (codebook
    training sampling); the packer streams per expert."""
    projections = _packed_expert_projection_names(profile, packed_proj)
    return torch.stack([_expert_weight(skeleton, profile, prefix, packed_proj,
                                       members, e)
                        for e in range(_n_experts(members, projections))])


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
    allow_route_pending_passthrough: bool = False,
    activation_cache_dir: str | Path | None = None,
    activation_scale_policy: str | None = None,
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

    ``reuse_prior`` is reserved but currently fails closed. The prior gate did
    not bind exact source content, treated an imatrix mismatch as a warning,
    sampled only some CB targets, and copied stock targets on dtype/shape alone.
    Reuse may return only after one immutable producer-input identity covers
    source bytes, imatrix, codebooks, scheme, and exporter ABI for every copied
    tensor. ``reuse_verify`` is retained only for CLI compatibility while reuse
    is blocked."""
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    if reuse_prior is not None:
        raise RuntimeError(
            "DELTA-EXPORT reuse is disabled: the prior artifact is not bound "
            "to exact source-content, imatrix, codebook, and exporter-ABI "
            "identity for every copied tensor. Re-encode into a fresh output "
            "directory."
        )
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

    assignment = load_assignment(layer_config_path)
    _recipe_payload = json.loads(Path(layer_config_path).read_text())
    _recipe_cb_context_stamp, _recipe_cb_tensor_stamps = (
        cb_serialization_metadata_from_assignment_payload(_recipe_payload)
    )
    _recipe_meta = _recipe_payload.get("__prismaquant__", {})
    _recipe_cb_render_identity = _recipe_payload.get("cb_render_identity")
    if _recipe_cb_render_identity is None and isinstance(_recipe_meta, dict):
        _recipe_cb_render_identity = _recipe_meta.get("cb_render_identity")
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
    expert_groups = _plan_expert_stacks(skeleton, profile)
    # The allocator writes its layer_config EXPANDED per tensor even though it
    # decided each expert group atomically, so a per-expert checkpoint arrives
    # as one entry per (expert, projection). Gridbook only names stacks. Do the
    # reduction here, once, before anything reads the assignment.
    assignment, expert_stack_members, expert_stack_report = (
        _collapse_per_expert_assignment(assignment, expert_groups, profile)
    )
    if expert_stack_members:
        col_weights = _packed_expert_col_weights(
            col_weights, expert_stack_members, profile)
        print(
            f"[export-cb-stream] collapsed "
            f"{expert_stack_report['members']} per-expert allocation entries "
            f"into {expert_stack_report['stacks']} packed expert stack(s)",
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

    def _base_name(qname: str) -> str:
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
            n = _n_experts(grp, projections)
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
    }
    activation_execution_contract = None
    activation_scales_by_physical_target: dict[str, float] = {}

    # --- Resolve/train codebooks (bounded pooling for learned). ---
    provided = spec.get("codebooks", {}) if source == "learned" else {}
    train = bool(spec.get("train", False))
    iters = int(spec.get("iters", 4))
    seed = int(spec.get("seed", 0))
    train_cap = int(spec.get("train_cap", 1 << 20))
    codebooks: dict[tuple[str, str], object] = {}
    target_cb: dict[str, tuple] = {}
    by_group: dict[tuple[str, str], list[str]] = {}
    for qname in cb_targets:
        fmt = assignment[qname]
        ref = _role_of(qname) if source == "learned" else "lattice"
        by_group.setdefault((ref, fmt), []).append(qname)
    for (ref, fmt), qnames in by_group.items():
        grid, mode, k = cb_targets[qnames[0]]
        if source == "lattice":
            codebooks[(ref, fmt)] = cb._resolve_codebook(
                k, grid, mode, None, torch.device(device))
            kind = "lattice"
        elif train:
            codebooks[(ref, fmt)] = _train_shared_codebook_streaming(
                skeleton, profile, expert_groups, _resolve_target,
                qnames, col_weights, grid=grid, mode=mode, k=k, seed=seed,
                iters=iters, train_cap=train_cap, device=device)
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
    materialized_codebook_digests = {
        name: hashlib.sha256(
            tensor.to(torch.float16).cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for name, tensor in materialized_codebook_tensors.items()
    }
    serialization_context = CBSerializationContext(
        scale_coding=scale_coding,
        codebook_source=source,
        scale_sweep=bool(scale_sweep),
        encode_tier=resolve_cb_encode_tier(),
        activation_contract=_claimed_activation_contract,
        activation_execution=(
            NVFP4_ACTIVATION_EXECUTION
            if _claimed_activation_contract is not None
            else None
        ),
        codebook_refs={
            qname: _codebook_tensor_names(ref, fmt, codebook)
            for qname, (ref, fmt, codebook, _kind) in target_cb.items()
        },
        codebook_content_digests=materialized_codebook_digests,
    )
    validate_cb_serialization_context_stamp(
        _recipe_cb_context_stamp,
        serialization_context,
        where="export_nvfp4_cb_streaming",
    )
    if cb_targets and _recipe_cb_render_identity is not None:
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
            where="export_nvfp4_cb_streaming assignment render identity",
        )
    elif cb_targets and production_recipe_stamped:
        raise ValueError(
            "export_nvfp4_cb_streaming: stamped production CB assignment is "
            "missing its value-bearing render identity"
        )
    elif cb_targets and not allow_unstamped_research:
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
                       (lambda k=wkey: skeleton.load(k).contiguous()))
            writer.add(
                export_base + ".weight_scale", torch.float32, ssh,
                (lambda k=skey: skeleton.load(k).to(torch.float32)
                 .contiguous()))
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
            packed, scale = _stream_pack_target(
                skeleton, profile, h, qname, grid, mode, k, codebook,
                col_weights[qname], scale_sweep, coding, shape, device,
                serialization_context.encode_tier,
                _recipe_cb_render_identity,
                verified_cb_source_qnames,
                expert_stack_members.get(qname))
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
                packed, scale = _stream_pack_target(
                    skeleton, profile, h, qname, grid, mode, k, codebook,
                    col_weights[qname], scale_sweep, coding, shape, device,
                    serialization_context.encode_tier,
                    _recipe_cb_render_identity,
                    verified_cb_source_qnames,
                    expert_stack_members.get(qname))
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

        specs = [
            spec
            for spec in _stock_output_specs(canon_fmt, shape)
            if not (
                qname in sidecar_stock
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
            for (name, dtype, out_shape), (suffix, _d, _o) in zip(
                    expected, specs):
                def _prod(suffix=suffix, _render=_render):
                    return _render()[suffix]
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
            for (name, dtype, out_shape), (suffix, _d, _o) in zip(
                    expected, specs):
                def _prod(suffix=suffix, _render=_render):
                    return _render()[suffix]
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
        # Consume only the source projections belonging to an actually-CB
        # packed parent. A partial layer (for example gate_up=CB, down=BF16)
        # must retain the untouched per-expert tensors for its BF16 parent.
        for packed_proj in _packed_expert_param_names(profile):
            checkpoint_qname = f"{prefix}.{packed_proj}"
            canon_qname = _canonical_qname(checkpoint_qname, profile)
            variants = {checkpoint_qname}
            if canon_qname is not None:
                variants.add(canon_qname)
            if not variants & cb_targets_set:
                continue
            for proj in _packed_expert_projection_names(profile, packed_proj):
                for base in projs.get(proj, {}).values():
                    consumed_expert_bases.add(base + ".weight")
    resolved_source_scale_keys = {
        entry[1] for entry in skeleton._fp8_scale_inv_map.values()
    }
    for name in skeleton.keys():
        if subset_prefixes is not None and \
                not any(name.startswith(p) for p in subset_prefixes):
            continue   # outside the declared subset (e.g. non-MTP body layers)
        if name in emitted_bases or name in consumed_expert_bases:
            continue
        if name in resolved_source_scale_keys or name.endswith(
            ".weight_scale_inv"
        ):
            continue   # consumed with its fp8 weight, or an unused sidecar
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
        writer.add(name, dtype, shape, (lambda k=name: skeleton.load(k)
                                        .contiguous()))
        if ckpt_qname is not None and len(shape) >= 2:
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
        return _base_name(qname)

    def _delegated_target_name(qname: str) -> str:
        return (
            profile.to_vllm_internal_name(qname)
            if profile is not None
            else qname
        )

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
        return _expert_group_of.get(qname, qname)

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
            unit = _decision_unit_id(qname)
            previous = units.setdefault(unit, fmt)
            if previous != fmt:
                raise ValueError(
                    f"{unit}: source-passthrough unit mixes {previous} and "
                    f"{fmt}; a unit ships on ONE contract or the export cannot "
                    "declare it")
        for prefix, projs in expert_groups.items():
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
                for name in tensor_names_by_target.get(qname, ())
            },
            "cb_modules": cb_modules,
            "passthrough_modules": passthrough_modules,
            "attested": set(routed_moe_attested_module_names(
                activation_execution_contract)),
        }

    _declared_passthrough_units = _source_passthrough_units()
    assert_routes_reconcile(
        **_route_reconciliation_sets(_declared_passthrough_units))

    quant_config = build_quant_config(
        assignment=assignment,
        cb_targets=cb_targets,
        source_targets=source_targets,
        native_source_targets=native_source_targets,
        requant_targets=requant_targets,
        stock_targets=stock_targets,
        by_group=by_group,
        codebooks=codebooks,
        col_weights=col_weights,
        codebook_tensors_by_name=cb_tensor_blobs,
        ignore=ignore,
        codebook_file=codebook_file,
        scale_coding=scale_coding,
        codebook_source=source,
        serialized_payload_summary=serialized_payload_summary,
        serialization_context=serialization_context,
        cb_render_identity=_recipe_cb_render_identity,
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
        route_pending_passthrough_acknowledged=sorted(route_pending),
        weight_only_stock_targets=sidecar_stock,
        streaming_provenance=True,
        include_tensor_formats=False,
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

    writer.write(
        out_dir / "model.safetensors",
        before_publish=_assert_source_coverage_before_publish,
    )
    if cb_tensor_blobs:
        save_file(cb_tensor_blobs, str(out_dir / codebook_file),
                  metadata={"format": "pt"})
    src_config = model_dir / "config.json"
    config = json.loads(src_config.read_text()) if src_config.exists() else {}
    config["quantization_config"] = {
        "quant_method": "gridbook", "format": "nvfp4_cb",
        "config_file": "quant_config.json"}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    for aux in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                "special_tokens_map.json", "generation_config.json",
                "vocab.json", "merges.txt", "chat_template.jinja",
                "chat_template.json", "preprocessor_config.json",
                "video_preprocessor_config.json", "processor_config.json"):
        p = model_dir / aux
        if p.exists():
            (out_dir / aux).write_bytes(p.read_bytes())
    # Persist and assert a final filesystem inventory distinct from the CB
    # tensor-data payload contract.  This includes both safetensors headers,
    # JSON configs, tokenizer assets, and all other regular output files.
    finalize_cb_export_artifact_inventory(
        out_dir,
        quant_config,
        serialized_payload=serialized_payload_summary,
        cb_tensor_names=sorted(cb_output_tensor_names),
        codebook_file=codebook_file,
        expected_model_files=["model.safetensors"],
        whole_artifact_budget_bytes=(
            int(_whole_artifact_budget["budget_bytes"])
            if _whole_artifact_budget is not None
            else None
        ),
    )
    return dict(counts)


def _stream_pack_target(skeleton, profile, resolved, qname, grid, mode, k,
                        codebook, cw, scale_sweep, coding, shape, device,
                        encode_tier, cb_render_identity,
                        verified_source_qnames, member_qnames=None):
    """Pack ONE target, streaming experts. Returns (packed uint8 (rows,bytes)
    or (E,out,bytes), scale-plane fp32 or None). Per-expert scales make
    per-expert packing byte-identical to whole-stack packing."""
    kind, h = resolved
    cbook = _to_device(codebook, device)
    if kind == "tensor":
        w = skeleton.dequant_weight(h).to(device)
        if cb_render_identity is not None:
            from prismaquant.production_weight_cache import (
                validate_cb_render_source_weight,
            )
            validate_cb_render_source_weight(
                cb_render_identity,
                qname,
                w,
                where="export_nvfp4_cb_streaming source tensor",
            )
            verified_source_qnames.add(qname)
        packed, fields = cb.nvfp4_cb_pack(
            w, k, grid=grid, mode=mode, col_weights=cw.to(device),
            codebook=cbook, scale_sweep=scale_sweep, scale_coding=coding,
            encode_tier=encode_tier)
        if w.dim() == 3:
            packed = packed.reshape(w.shape[0], w.shape[1], -1)
        scale = (fields["scales"].reshape(*w.shape[:-1]).cpu()
                 if grid == "fp8" else None)
        return packed.to(torch.uint8).cpu().contiguous(), scale
    # Experts: build ONE layer's stack (fp4 derives a single per-tensor global
    # over the whole stack, so per-expert packing would diverge — the stack is
    # the byte-identity working set) and pack it whole, exactly as the
    # in-memory exporter packs a pre-stacked 3-D tensor. Peak = one MoE layer's
    # experts, not the model.
    prefix, packed_proj, grp = h
    projections = _packed_expert_projection_names(profile, packed_proj)
    n = _n_experts(grp, projections)
    on_member = None
    if cb_render_identity is not None and member_qnames is not None:
        # A per-expert checkpoint's render identity is keyed PER EXPERT (the
        # cost rows and the imatrix are too), so the stack is verified member
        # by member as it is decoded. Hashing the concatenated stack instead
        # would certify a name the recipe never priced.
        from prismaquant.production_weight_cache import (
            validate_cb_render_source_weight,
        )

        def on_member(proj, e, _base, decoded, _q=member_qnames):
            member = _q[(proj, e)]
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
    first = _expert_weight(skeleton, profile, prefix, packed_proj, grp, 0,
                           on_member=on_member)
    w = torch.empty((n, *first.shape), dtype=first.dtype, device=device)
    w[0] = first.to(device)
    del first
    for e in range(1, n):
        chunk = _expert_weight(skeleton, profile, prefix, packed_proj, grp, e,
                               on_member=on_member)
        w[e] = chunk.to(device)
        del chunk
    if cb_render_identity is not None and member_qnames is None:
        from prismaquant.production_weight_cache import (
            validate_cb_render_source_weight,
        )
        validate_cb_render_source_weight(
            cb_render_identity,
            qname,
            w,
            where="export_nvfp4_cb_streaming expert stack",
        )
        verified_source_qnames.add(qname)
    packed, fields = cb.nvfp4_cb_pack(
        w, k, grid=grid, mode=mode, col_weights=cw.to(device),
        codebook=cbook, scale_sweep=scale_sweep, scale_coding=coding,
        encode_tier=encode_tier)
    packed = packed.reshape(w.shape[0], w.shape[1], -1)
    scale = (fields["scales"].reshape(*w.shape[:-1]).cpu()
             if grid == "fp8" else None)
    return packed.to(torch.uint8).cpu().contiguous(), scale


def _train_shared_codebook_streaming(skeleton, profile, expert_groups,
                                     resolve_target, qnames, col_weights, *,
                                     grid, mode, k, seed, iters, train_cap,
                                     device):
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
            weights.append(_stacked_source_weight(
                skeleton, profile, prefix, packed_proj, grp).to(device))
        cws.append(col_weights[q].to(device))
    return _train_shared_codebook(
        weights, cws, grid=grid, mode=mode, k=k, seed=seed, iters=iters,
        train_cap=train_cap)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Streaming NVFP4-CB exporter")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer-config", required=True)
    ap.add_argument("--out", required=True)
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
        "--allow-route-pending-passthrough",
        action="store_true",
        help="ship a source-passthrough rung whose serve route is not yet "
        "validated (allocator_candidates.ROUTE_PENDING_PASSTHROUGH_FORMATS, "
        "today MXFP4_SOURCE -> lane delegated_native_mxfp4). Refused by "
        "default; the acknowledgement is recorded in the artifact provenance.",
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
    ap.add_argument("--reuse-prior", default=None, metavar="DIR",
                    help="reserved DELTA-EXPORT input; currently fails closed "
                         "until exact source/imatrix/codebook/exporter identity "
                         "is implemented. Env PRISMAQUANT_EXPORT_REUSE_PRIOR "
                         "is also rejected.")
    ap.add_argument("--reuse-verify", type=int, default=None, metavar="N",
                    help="reserved compatibility option while DELTA-EXPORT is "
                         "disabled (default 3; env "
                         "PRISMAQUANT_EXPORT_REUSE_VERIFY)")
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
        allow_route_pending_passthrough=args.allow_route_pending_passthrough,
        activation_cache_dir=args.activation_cache_dir,
        activation_scale_policy=args.activation_scale_policy)
    size = sum(p.stat().st_size for p in Path(args.out).glob("*")) / 1e9
    print(f"wrote {args.out} ({size:.3f} GB)")
    for fmt, n in sorted(counts.items()):
        print(f"  {fmt}: {n}")


if __name__ == "__main__":
    main()
