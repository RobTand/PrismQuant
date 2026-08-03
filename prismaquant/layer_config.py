"""Shared layer-config parsing helpers.

PrismaQuant writes allocator assignments in a few shapes: shorthand strings,
integer bit widths, and AutoRound-style dictionaries. Keep the production
parser in one place so export, recache, KL validation, and small tools cannot
silently disagree about the same recipe.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from prismaquant.cb_layout import CB_FORMAT_NAMES
from prismaquant.schemas import validate_layer_config_payload


def strip_weight(name: str) -> str:
    """Normalize tensor names to module qnames."""
    return name[:-len(".weight")] if name.endswith(".weight") else name


# Reserved non-tensor key in layer_config.json. No module qname can collide
# with it (dunder-wrapped, no dots), so allocator metadata can travel WITH the
# assignment instead of in a side report the exporter never reads (re-vet R11 /
# debt D4: the allocator resolved `vllm_packed_moe` while export re-resolved
# `gguf` from the spec, silently coercing 226 Hy3 FP8 Linears to BF16).
LAYER_CONFIG_META_KEY = "__prismaquant__"


def is_layer_config_meta_key(name: str) -> bool:
    return str(name) == LAYER_CONFIG_META_KEY


def layer_config_metadata(payload: Mapping) -> dict:
    """Return the reserved metadata block of a layer_config payload."""
    meta = payload.get(LAYER_CONFIG_META_KEY) if isinstance(payload, Mapping) else None
    return dict(meta) if isinstance(meta, Mapping) else {}


def read_layer_config_metadata(path: str | Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return {}
    return layer_config_metadata(payload) if isinstance(payload, Mapping) else {}


# GGUF k-quant + IQ lane (llama.cpp / vLLM-GGUF serving). Kept as an explicit
# literal so this module stays torch-free; pinned to gguf_formats.GGUF_BLOCK_BYTES
# by test_gguf_formats.test_layer_config_gguf_names_stay_in_sync.
_GGUF_FORMAT_NAMES = frozenset(
    {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0",
     "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL"}
)

# Backwards-compatible private name used by exporters. The canonical producer
# ladder is torch-free ``cb_layout.CB_FORMAT_NAMES``; do not rebuild it here.
_NVFP4_CB_FORMAT_NAMES = CB_FORMAT_NAMES

# Checkpoint ``quantization_config.scale_fmt`` spellings that mean a one-byte
# UE8M0 block exponent. Kept as a literal so this module stays torch-free;
# ``layer_streaming._E8M0_SCALE_FMTS`` is the decode-side twin.
_UE8M0_SCALE_FMTS = frozenset({"ue8m0", "e8m0"})


def canonicalize_format(entry: dict | str | int) -> str:
    """Map a layer-config entry to an export/runtime format name.

    This parser is runtime-neutral: research formats such as E5M2 are
    canonicalized here, then serving/export profiles decide whether they are
    legal for a concrete backend.
    """
    if isinstance(entry, dict):
        dt = entry.get("data_type")
        bits = int(entry.get("bits", 0))
        if dt == "gguf":
            gguf_type = str(entry.get("gguf_type", "")).upper()
            if gguf_type not in _GGUF_FORMAT_NAMES:
                raise ValueError(f"unsupported gguf scheme: {entry!r}")
            return gguf_type
        if dt == "nvfp4_cb":
            rung = "S" if str(entry.get("cb_mode", "")) == "signed" else "K"
            name = f"NVFP4_CB_{rung}{int(entry['cb_k'])}"
            if name not in _NVFP4_CB_FORMAT_NAMES:
                raise ValueError(f"unsupported nvfp4_cb scheme: {entry!r}")
            return name
        if dt == "fp8_cb":
            name = f"FP8_CB_K{int(entry['cb_k'])}"
            if name not in _NVFP4_CB_FORMAT_NAMES:
                raise ValueError(f"unsupported fp8_cb scheme: {entry!r}")
            return name
        if dt == "nv_fp" and bits == 4:
            return "NVFP4"
        if dt == "mx_fp" and bits == 4:
            return "MXFP4"
        if dt == "fp4_e2m1" and bits == 4:
            # MXFP4_SOURCE — the byte-verbatim OCP-MX passthrough, named by
            # its ELEMENT dtype rather than by ``mx_fp`` precisely so it does
            # not collide with MXFP4 (the rung this pipeline would re-encode
            # itself). The distinction is not cosmetic: MXFP4 is a format the
            # exporter produces, MXFP4_SOURCE is a claim that the checkpoint
            # already ships these exact bytes and the exporter must copy them.
            # group_size is part of the claim (OCP-MX blocks are 32), so a
            # different one is a different contract, not a variant.
            group_size = int(entry.get("group_size", 0))
            if group_size != 32:
                raise ValueError(
                    "MXFP4_SOURCE is the OCP-MX group-of-32 passthrough; "
                    f"got group_size={group_size} in {entry!r}"
                )
            return "MXFP4_SOURCE"
        if dt == "mx_fp" and bits == 8:
            elt = str(entry.get("weight_element_dtype", "fp8_e4m3")).lower()
            if elt == "fp8_e5m2":
                return "MXFP8_E5M2"
            return "MXFP8_E4M3"
        if dt in ("float", "bfloat16") and bits in (16, 0):
            return "BF16"
        if dt == "fp8_e4m3" and bits == 8:
            group_size = int(entry.get("group_size", 0))
            if group_size == 128:
                # Same E4M3 element grid and same 128x128 block, but the SCALE
                # PLANE differs and that is the whole on-disk contract: a
                # one-byte UE8M0 exponent (DeepSeek-V3.1/V4) is not the FP32
                # weight_scale_inv plane FP8_SOURCE names, and the exporter
                # widens the latter to FP32 on write. Reading them as one
                # format would ship 4x the scale bytes in a layout the
                # checkpoint's own loader does not expect.
                if str(entry.get("scale_fmt", "")).lower() in _UE8M0_SCALE_FMTS:
                    return "FP8_BLOCK_UE8M0_SOURCE"
                return "FP8_SOURCE"
            if group_size == 32:
                # Same split as the group-128 case immediately above, one
                # granularity down: an explicit UE8M0 scale_fmt names the
                # Gridbook-native rung whose scale plane is float8_e8m0fnu and
                # whose encoder is the saturating-ceil rule, NOT the stock
                # compressed-tensors MXFP8 scheme (uint8 scales, OCP rule).
                # Absent scale_fmt keeps the historical reading.
                if str(entry.get("scale_fmt", "")).lower() in _UE8M0_SCALE_FMTS:
                    return "MXFP8_UE8M0_G32"
                return "MXFP8_E4M3"
            if group_size in (0, -1):
                return "FP8_E4M3"
            return "MXFP8_E4M3"
        if dt == "fp8_e5m2" and bits == 8:
            return "FP8_E5M2"
        if dt == "mx_fp" and bits == 6:
            elt = str(entry.get("weight_element_dtype", "fp6_e3m2")).lower()
            if elt == "fp6_e2m3":
                return "MXFP6_E2M3"
            return "MXFP6_E3M2"
        if dt == "fp6_e3m2" and bits == 6:
            return "MXFP6_E3M2"
        if dt == "fp6_e2m3" and bits == 6:
            return "MXFP6_E2M3"
        raise ValueError(f"unsupported scheme: {entry!r}")
    if isinstance(entry, str):
        value = entry.lower()
        if value.upper() in _GGUF_FORMAT_NAMES:
            return value.upper()
        if value.upper() in _NVFP4_CB_FORMAT_NAMES:
            return value.upper()
        if value in ("nvfp4", "fp4", "4"):
            return "NVFP4"
        if value in ("mxfp4_source", "mx_fp4_source"):
            # Checked BEFORE the plain MXFP4 spellings: a substring-free
            # equality test would still pass either way, but keeping the
            # narrower claim first documents that the two are different
            # contracts rather than aliases.
            return "MXFP4_SOURCE"
        if value in ("mxfp4", "mx_fp4"):
            return "MXFP4"
        if value in ("mxfp8_ue8m0_g32", "mxfp8_ue8m0"):
            # Checked BEFORE the plain MXFP8 spellings, same reason as
            # MXFP4_SOURCE above: two different on-disk contracts, and the
            # narrower claim reads first so the pair is obviously deliberate.
            return "MXFP8_UE8M0_G32"
        if value in ("mxfp8", "mxfp8_e4m3"):
            return "MXFP8_E4M3"
        if value in ("fp8_block_ue8m0_source", "fp8_block_ue8m0"):
            return "FP8_BLOCK_UE8M0_SOURCE"
        if value in ("fp8_source", "fp8_block_source"):
            return "FP8_SOURCE"
        if value in ("fp8", "fp8_dynamic", "fp8_e4m3", "fp8_e4m3fn", "8"):
            return "FP8_E4M3"
        if value in ("mxfp8_e5m2", "mx_fp8_e5m2"):
            return "MXFP8_E5M2"
        if value in ("fp8_e5m2", "fp8_e5m2fn"):
            return "FP8_E5M2"
        if value in ("bf16", "bfloat16", "16"):
            return "BF16"
        raise ValueError(f"unsupported format string: {entry!r}")
    if isinstance(entry, int):
        if entry <= 4:
            return "NVFP4"
        if entry <= 8:
            return "FP8_E4M3"
        return "BF16"
    raise ValueError(f"unsupported scheme: {entry!r}")


def canonicalize_assignment(raw: Mapping) -> dict[str, str]:
    return {
        strip_weight(str(name)): canonicalize_format(entry)
        for name, entry in raw.items()
        if not is_layer_config_meta_key(name)
    }


def load_assignment(path: str | Path) -> dict[str, str]:
    path = Path(path)
    payload = json.loads(path.read_text())
    validate_layer_config_payload(payload, str(path))
    return canonicalize_assignment(payload)
