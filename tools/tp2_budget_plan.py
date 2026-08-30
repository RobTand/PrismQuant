#!/usr/bin/env python3
"""Two-box (DGX Spark cluster) residency budget planner for a shipped artifact.

The question this tool settles: given an artifact and a parallelism spec,
what actually RESIDES on each rank, and how large an artifact still fits?

It is a *residency* planner, not a speed model.  It never loads a tensor: it
reads safetensors headers only (the 8-byte length prefix plus the JSON header)
and computes per-tensor bytes from dtype x shape, cross-checked against the
header's own data_offsets.  That cross-check is what earns the EXACT label.

Honesty contract (ported from tools/tp_decode_feasibility.py):
  * Every number carries a provenance label.  EXACT means byte math off the
    checkpoint.  DERIVED means computed from a rule read out of another
    codebase (vLLM's layer split).  ASSUMED means a modelling choice this
    tool made and could be wrong about.  MEASURED is reserved for numbers
    that came from an instrument.
  * Nothing is silently bucketed.  A tensor matching no classification rule
    is a hard error that lists the offending names (the silent-zero lesson:
    silent buckets rank broken arms first).
  * TP mode does not pretend to know the shard policy.  It prints an ASSUMED
    all-body-sharded model, labels every line of that table ASSUMED, and says
    what would have to be attested to remove the label.
  * Missing overhead is printed as UNKNOWN-NEEDS-OVERHEAD, never defaulted to
    a number (same discipline as UNKNOWN-NEEDS-TPOT in tp_decode_feasibility).

Checkpoint-name classification (checkpoint namespace ONLY -- never the recipe
or vLLM-internal namespace):

  1. sidecar   any dot-segment that is or starts with mtp/nextn/draft.
               Checked FIRST, so an MTP block that carries its own
               ``.layers.N.`` names cannot be mistaken for a body layer and
               ``mtp.norm.weight`` cannot be mistaken for the final norm.
  2. layer     ``(?:^|\\.)layers\\.(\\d+)\\.``  -- note the ``^|`` extension
               over a bare ``\\.layers\\.``: DSv4 checkpoints name their body
               ``layers.0.attn...`` with no ``model.`` prefix, and the bare
               form matches none of them.
  3. embed     first segment (after stripping model./language_model./
               transformer. wrappers) in {embed_tokens, embed, wte,
               tok_embeddings}  -> FIRST stage.
  4. head      that first segment == lm_head or head (EXACT rule), or merely
               CONTAINS "head" (e.g. DSv4's ``hc_head_base``) -- the latter is
               line-itemed ASSUMED-PLACEMENT because this tool does not know
               that such a head is last-stage-resident only.  -> LAST stage.
  5. norm      that first segment in {norm, final_norm, final_layernorm,
               ln_f}  -> LAST stage.

Anything else: hard error.

Non-safetensors payload files (codebooks, quant_config.json, ...) are
line-itemed under REPORTED-NOT-CLASSIFIED with their on-disk size and are
EXCLUDED from every rank-residency number.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "prismaquant.tp2_budget_plan.v1"

GIB = 1024 ** 3
GB_UNIT_NOTE = "all *GB columns are GiB = 1024**3 bytes"

# Provenance labels.  EXACT / DERIVED / ASSUMED / MEASURED is load-bearing.
LABEL_EXACT = "EXACT / safetensors header (dtype x shape, offset-verified)"
LABEL_DERIVED_VLLM = "DERIVED / vllm get_pp_indices default split"
LABEL_EXACT_DP = "EXACT / balanced contiguous split (DP over exact bytes)"
LABEL_ASSUMED_TP = "ASSUMED / no shard-policy source; see TP CAVEAT"
LABEL_ASSUMED_LINEAR = "ASSUMED-LINEAR-SCALING"
LABEL_ASSUMED_PLACEMENT = "ASSUMED-PLACEMENT"
UNKNOWN_OVERHEAD = "UNKNOWN-NEEDS-OVERHEAD"

# The default is the operator's rule of thumb for a 128 GB DGX Spark
# (MemTotal on sparky is 121.6 GiB); it is NOT a measurement made here.
PER_DEVICE_GB_DEFAULT = 121.0
LABEL_PER_DEVICE_DEFAULT = (
    "ASSUMED / default 121 GiB usable of a 128 GB unified pool (not measured "
    "by this tool; pass --per-device-gb to override)")

# Measured cross-box link facts, printed for context only.  This planner does
# not model bandwidth or latency -- tools/tp_decode_feasibility.py does.
LINK_JSON_DEFAULT = Path("/home/rob/dq-runs/tp2-2026-08-23/nccl_allreduce.json")

# safetensors dtype -> bytes per element.  An unknown dtype is a hard error,
# never a guess: sub-byte dtypes would silently halve the artifact.
DTYPE_BYTES: Dict[str, int] = {
    "BOOL": 1,
    "U8": 1, "I8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
SIDECAR_SEGMENT_RE = re.compile(r"^(mtp|nextn|draft)([_.\-].*)?$")
WRAPPER_PREFIXES = ("model.language_model.", "language_model.", "model.",
                    "transformer.")
EMBED_SEGMENTS = {"embed_tokens", "embed", "wte", "tok_embeddings"}
HEAD_SEGMENTS = {"lm_head", "head"}
NORM_SEGMENTS = {"norm", "final_norm", "final_layernorm", "ln_f"}

BUCKET_LAYER = "layer"
BUCKET_EMBED = "embed"
BUCKET_HEAD = "head"
BUCKET_NORM = "final_norm"
BUCKET_SIDECAR = "sidecar"

METADATA_EXTENSIONS = {".json", ".txt", ".md", ".jinja", ".png", ".model",
                       ".py", ".yaml", ".yml"}


class ClassificationError(RuntimeError):
    """A tensor matched no classification rule (fail closed, never bucket it)."""


class HeaderError(RuntimeError):
    """The safetensors header is unreadable or internally inconsistent."""


# ---------------------------------------------------------------------------
# Pure byte math and classification (unit-tested; no artifact needed).


def dtype_bytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise HeaderError(
            f"unknown safetensors dtype {dtype!r}; refusing to guess its width "
            f"(known: {', '.join(sorted(DTYPE_BYTES))})") from None


def tensor_bytes(dtype: str, shape: Sequence[int]) -> int:
    """Exact byte size from dtype x shape.  A 0-d tensor is one element."""
    n = 1
    for dim in shape:
        if int(dim) < 0:
            raise HeaderError(f"negative dimension in shape {list(shape)}")
        n *= int(dim)
    return n * dtype_bytes(dtype)


def read_safetensors_header(path: Path) -> Dict[str, Any]:
    """Parse the 8-byte-length JSON header only.  Tensor data is never read."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise HeaderError(f"{path}: file shorter than the 8-byte header length")
        (hlen,) = struct.unpack("<Q", raw)
        if hlen <= 0 or hlen > 512 * 1024 * 1024:
            raise HeaderError(f"{path}: implausible safetensors header length {hlen}")
        blob = fh.read(hlen)
        if len(blob) != hlen:
            raise HeaderError(f"{path}: truncated safetensors header")
    try:
        header = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise HeaderError(f"{path}: safetensors header is not JSON ({exc})") from exc
    if not isinstance(header, dict):
        raise HeaderError(f"{path}: safetensors header is not a JSON object")
    return header


def entries_from_header(header: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
    """Header dict -> per-tensor entries, with the offset cross-check applied.

    The cross-check (dtype x shape == data_offsets delta) is what makes the
    EXACT label honest; a mismatch means the dtype table is wrong for this
    checkpoint and the whole plan would be wrong with it.
    """
    out: List[Dict[str, Any]] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict) or "dtype" not in meta or "shape" not in meta:
            raise HeaderError(f"{source}: tensor {name!r} has no dtype/shape metadata")
        nbytes = tensor_bytes(meta["dtype"], meta["shape"])
        offsets = meta.get("data_offsets")
        if isinstance(offsets, (list, tuple)) and len(offsets) == 2:
            span = int(offsets[1]) - int(offsets[0])
            if span != nbytes:
                raise HeaderError(
                    f"{source}: tensor {name!r} dtype x shape = {nbytes} B but "
                    f"data_offsets span {span} B; the dtype width table does "
                    f"not describe this checkpoint")
        out.append({
            "name": name,
            "dtype": meta["dtype"],
            "shape": [int(d) for d in meta["shape"]],
            "bytes": nbytes,
            "file": source,
        })
    return out


def _endcap_segment(name: str) -> str:
    """First segment after stripping model./language_model./transformer."""
    stripped = name
    changed = True
    while changed:
        changed = False
        for prefix in WRAPPER_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                changed = True
    return stripped.split(".", 1)[0]


def classify(name: str) -> Dict[str, Any]:
    """Classify one checkpoint tensor name.  Raises ClassificationError."""
    segments = name.split(".")
    if any(SIDECAR_SEGMENT_RE.match(seg) for seg in segments):
        return {"bucket": BUCKET_SIDECAR, "layer": None, "rule": "sidecar segment",
                "assumed_placement": False}
    m = LAYER_RE.search(name)
    if m:
        return {"bucket": BUCKET_LAYER, "layer": int(m.group(1)),
                "rule": "layers.<n>", "assumed_placement": False}
    seg = _endcap_segment(name)
    if seg in EMBED_SEGMENTS:
        return {"bucket": BUCKET_EMBED, "layer": None, "rule": f"embed segment {seg!r}",
                "assumed_placement": False}
    if seg in HEAD_SEGMENTS:
        return {"bucket": BUCKET_HEAD, "layer": None, "rule": f"head segment {seg!r}",
                "assumed_placement": False}
    if seg in NORM_SEGMENTS:
        return {"bucket": BUCKET_NORM, "layer": None, "rule": f"norm segment {seg!r}",
                "assumed_placement": False}
    if "head" in seg:
        return {"bucket": BUCKET_HEAD, "layer": None,
                "rule": f"segment {seg!r} contains 'head'", "assumed_placement": True}
    raise ClassificationError(name)


def classify_entries(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Classify every entry; one hard error listing ALL unmatched names."""
    out: List[Dict[str, Any]] = []
    unmatched: List[str] = []
    for e in entries:
        try:
            cls = classify(e["name"])
        except ClassificationError:
            unmatched.append(e["name"])
            continue
        out.append({**e, **cls})
    if unmatched:
        listed = "\n  ".join(sorted(unmatched))
        raise ClassificationError(
            f"{len(unmatched)} tensor name(s) matched no pipeline-stage rule; "
            f"refusing to plan with a silent bucket:\n  {listed}")
    return out


def layer_byte_table(classified: Sequence[Dict[str, Any]]) -> List[int]:
    """Per-layer byte totals indexed 0..L-1.  Layer ids must be contiguous."""
    per: Dict[int, int] = {}
    for e in classified:
        if e["bucket"] == BUCKET_LAYER:
            per[e["layer"]] = per.get(e["layer"], 0) + e["bytes"]
    if not per:
        raise ClassificationError("no decoder-layer tensors found in this artifact")
    ids = sorted(per)
    if ids != list(range(len(ids))):
        raise ClassificationError(
            f"decoder layer ids are not contiguous 0..L-1: found {ids}")
    return [per[i] for i in ids]


def bucket_bytes(classified: Sequence[Dict[str, Any]], bucket: str) -> int:
    return sum(e["bytes"] for e in classified if e["bucket"] == bucket)


# ---------------------------------------------------------------------------
# Splits.


def vllm_even_split(num_layers: int, pp_size: int) -> List[int]:
    """vLLM's default layer partition (get_pp_indices, no VLLM_PP_LAYER_PARTITION).

    Transcribed from vllm/distributed/utils.py::get_pp_indices: the remainder
    is added to partitions[-2], [-3], ... -- the LAST rank is skipped because
    it carries the output endcaps.  43 layers at pp:2 -> [22, 21].
    """
    if pp_size < 1:
        raise ValueError("pp_size must be >= 1")
    if num_layers < pp_size:
        raise ValueError(
            f"{num_layers} decoder layers cannot fill {pp_size} pipeline stages")
    base = num_layers // pp_size
    partitions = [base] * pp_size
    remaining = num_layers % pp_size
    for i in range(2, remaining + 2):
        partitions[-i] += 1
    return partitions


def split_to_ranges(counts: Sequence[int]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    for c in counts:
        ranges.append((start, start + c))
        start += c
    return ranges


def rank_weight_bytes(
    counts: Sequence[int],
    layer_bytes: Sequence[int],
    first_extra: int,
    last_extra: int,
) -> List[int]:
    """Resident weight bytes per rank for a contiguous layer split."""
    out: List[int] = []
    for r, (lo, hi) in enumerate(split_to_ranges(counts)):
        total = sum(layer_bytes[lo:hi])
        if r == 0:
            total += first_extra
        if r == len(counts) - 1:
            total += last_extra
        out.append(total)
    return out


def optimal_split(
    layer_bytes: Sequence[int],
    pp_size: int,
    first_extra: int = 0,
    last_extra: int = 0,
) -> List[int]:
    """Contiguous split minimising the maximum per-rank resident bytes.

    Exact DP (layer counts are small): dp[r][i] = the best achievable maximum
    when the first r ranks cover layers [0, i).  Ties are broken toward the
    EARLIEST boundary, so the answer is deterministic.
    """
    n = len(layer_bytes)
    if pp_size < 1:
        raise ValueError("pp_size must be >= 1")
    if n < pp_size:
        raise ValueError(
            f"{n} decoder layers cannot fill {pp_size} pipeline stages")
    prefix = [0] * (n + 1)
    for i, b in enumerate(layer_bytes):
        prefix[i + 1] = prefix[i] + b

    def span(lo: int, hi: int, rank: int) -> int:
        total = prefix[hi] - prefix[lo]
        if rank == 0:
            total += first_extra
        if rank == pp_size - 1:
            total += last_extra
        return total

    inf = math.inf
    # dp[r][i]: r ranks used, covering layers [0, i)
    dp: List[List[float]] = [[inf] * (n + 1) for _ in range(pp_size + 1)]
    choice: List[List[int]] = [[-1] * (n + 1) for _ in range(pp_size + 1)]
    dp[0][0] = 0.0
    for r in range(1, pp_size + 1):
        for i in range(r, n - (pp_size - r) + 1):
            best = inf
            best_j = -1
            for j in range(r - 1, i):
                if dp[r - 1][j] == inf:
                    continue
                cand = max(dp[r - 1][j], float(span(j, i, r - 1)))
                if cand < best:
                    best = cand
                    best_j = j
            dp[r][i] = best
            choice[r][i] = best_j
    if dp[pp_size][n] == inf:  # pragma: no cover - guarded by the n < pp_size check
        raise ValueError("no feasible contiguous split")
    counts: List[int] = []
    i = n
    for r in range(pp_size, 0, -1):
        j = choice[r][i]
        counts.append(i - j)
        i = j
    counts.reverse()
    return counts


# ---------------------------------------------------------------------------
# Feasibility arithmetic.


def max_feasible_artifact_bytes(
    counts: Sequence[int],
    layer_bytes: Sequence[int],
    first_extra: int,
    last_extra: int,
    budget_bytes: float,
    overhead_bytes: float,
) -> Dict[str, Any]:
    """Largest artifact that still fits, under ASSUMED-LINEAR-SCALING.

    Two assumptions, both printed by the caller:
      (1) body bytes scale linearly with artifact size,
      (2) each rank keeps the body SHARE it holds under the chosen split at
          today's size (shares frozen; the split is not re-solved),
      and the endcaps (embed / head / final norm / sidecar) stay fixed.

    Then rank r holds  s * body_total * f_r + endcap_r + overhead, and
    s_max = min_r (budget - overhead - endcap_r) / (body_total * f_r).
    """
    body_total = sum(layer_bytes)
    endcaps = [0] * len(counts)
    endcaps[0] += first_extra
    endcaps[-1] += last_extra
    shares: List[float] = []
    for lo, hi in split_to_ranges(counts):
        shares.append((sum(layer_bytes[lo:hi]) / body_total) if body_total else 0.0)
    limits: List[Optional[float]] = []
    binding = None
    s_max = math.inf
    for r, share in enumerate(shares):
        avail = budget_bytes - overhead_bytes - endcaps[r]
        if avail <= 0:
            return {
                "feasible": False,
                "reason": (f"rank {r} endcaps ({endcaps[r]} B) plus overhead "
                           f"already exceed the per-device budget"),
                "s_max": 0.0,
                "max_artifact_bytes": 0,
                "binding_rank": r,
                "rank_limits": limits,
            }
        if share <= 0:
            limits.append(None)
            continue
        limit = avail / (body_total * share)
        limits.append(limit)
        if limit < s_max:
            s_max = limit
            binding = r
    fixed = first_extra + last_extra
    return {
        "feasible": True,
        "reason": None,
        "s_max": s_max,
        "body_total_bytes": body_total,
        "fixed_endcap_bytes": fixed,
        "max_body_bytes": s_max * body_total,
        "max_artifact_bytes": s_max * body_total + fixed,
        "binding_rank": binding,
        "rank_limits": limits,
    }


# ---------------------------------------------------------------------------
# TP (assumed) model.


def tp_assumed_replicated(name: str, bucket: str) -> bool:
    """ASSUMED shard policy: norms, biases and the embed/head endcaps replicate.

    This is a modelling choice, not knowledge.  The real replicated set is a
    property of the serving runtime's parallel-Linear registration, which this
    producer-side tool has no attested source for.
    """
    if bucket in (BUCKET_EMBED, BUCKET_HEAD, BUCKET_NORM):
        return True
    segments = name.split(".")
    if any("norm" in seg for seg in segments):
        return True
    if segments[-1] == "bias" or any(seg == "bias" for seg in segments):
        return True
    return False


def tp_plan(classified: Sequence[Dict[str, Any]], tp_size: int) -> Dict[str, Any]:
    replicated = 0
    sharded = 0
    for e in classified:
        if tp_assumed_replicated(e["name"], e["bucket"]):
            replicated += e["bytes"]
        else:
            sharded += e["bytes"]
    per_rank = sharded / tp_size + replicated
    return {
        "tp_size": tp_size,
        "assumed_replicated_bytes": replicated,
        "assumed_sharded_bytes": sharded,
        "assumed_per_rank_bytes": per_rank,
        "label": LABEL_ASSUMED_TP,
    }


# ---------------------------------------------------------------------------
# Artifact IO.


def parse_parallelism(spec: str) -> Tuple[str, int]:
    if ":" not in spec:
        raise SystemExit(f"--parallelism must look like pp:2 or tp:2 (got {spec!r})")
    mode, _, count = spec.partition(":")
    mode = mode.strip().lower()
    if mode not in ("pp", "tp"):
        raise SystemExit(f"--parallelism mode must be pp or tp (got {mode!r})")
    try:
        n = int(count)
    except ValueError:
        raise SystemExit(f"--parallelism size must be an integer (got {count!r})") from None
    if n < 2:
        raise SystemExit(f"--parallelism size must be >= 2 (got {n})")
    return mode, n


def collect_artifact_tensors(artifact_dir: Path) -> Dict[str, Any]:
    """Read every safetensors header in the artifact (index-aware)."""
    index_path = artifact_dir / "model.safetensors.index.json"
    single = artifact_dir / "model.safetensors"
    entries: List[Dict[str, Any]] = []
    index_meta: Optional[Dict[str, Any]] = None
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise HeaderError(f"{index_path}: no usable weight_map")
        shard_files = sorted(set(weight_map.values()))
        for shard in shard_files:
            shard_path = artifact_dir / shard
            if not shard_path.exists():
                raise HeaderError(f"{index_path}: shard {shard} referenced but missing")
            entries.extend(entries_from_header(
                read_safetensors_header(shard_path), shard))
        header_names = {e["name"] for e in entries}
        map_names = set(weight_map)
        missing = sorted(map_names - header_names)
        extra = sorted(header_names - map_names)
        if missing or extra:
            raise HeaderError(
                f"{index_path}: weight_map and shard headers disagree; "
                f"{len(missing)} in index only (e.g. {missing[:5]}), "
                f"{len(extra)} in shards only (e.g. {extra[:5]})")
        index_meta = {
            "shards": shard_files,
            "declared_total_size": index.get("metadata", {}).get("total_size"),
        }
        declared = index_meta["declared_total_size"]
        computed = sum(e["bytes"] for e in entries)
        if declared is not None and int(declared) != computed:
            raise HeaderError(
                f"{index_path}: metadata.total_size {declared} != computed "
                f"tensor bytes {computed}")
        source = "sharded index"
    elif single.exists():
        entries = entries_from_header(read_safetensors_header(single),
                                      "model.safetensors")
        source = "single model.safetensors (no index)"
    else:
        raise HeaderError(
            f"{artifact_dir}: neither model.safetensors.index.json nor "
            f"model.safetensors is present")
    names = [e["name"] for e in entries]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise HeaderError(f"{artifact_dir}: duplicate tensor names {dupes[:5]}")
    return {"entries": entries, "source": source, "index": index_meta}


def unclassified_payload_files(artifact_dir: Path) -> List[Dict[str, Any]]:
    """Every non-safetensors file, by size.  No threshold, no bucketing."""
    rows: List[Dict[str, Any]] = []
    for path in sorted(artifact_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            continue
        if path.name == "model.safetensors.index.json":
            continue
        rows.append({
            "file": path.name,
            "bytes": path.stat().st_size,
            "kind": "metadata-extension" if path.suffix in METADATA_EXTENSIONS
                    else "payload",
        })
    rows.sort(key=lambda r: -r["bytes"])
    return rows


# ---------------------------------------------------------------------------
# Rendering.


def gib(nbytes: float) -> str:
    return f"{nbytes / GIB:.3f}"


def check_out_path(path: Path) -> Path:
    """Refuse host /tmp for outputs (standing project rule: /tmp gets cleared)."""
    resolved = Path(path).expanduser().resolve()
    if resolved == Path("/tmp") or Path("/tmp") in resolved.parents:
        raise SystemExit(
            f"refusing to write under /tmp ({resolved}); /tmp was cleared by an "
            f"OOM once already and took artifacts with it")
    return resolved


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _link_facts(path: Path) -> Optional[Dict[str, Any]]:
    try:
        blob = json.loads(Path(path).read_text())
    except Exception:
        return None
    rows = blob.get("rows") or []
    if not rows:
        return None
    small = min(rows, key=lambda r: r["bytes"])
    fastest = max(rows, key=lambda r: r.get("busbw_GBps", 0.0))
    return {
        "source": str(path),
        "provenance": blob.get("provenance"),
        "small_msg_bytes": small["bytes"],
        "small_msg_p50_us": small["p50_us"],
        "peak_busbw_GBps": fastest.get("busbw_GBps"),
        "peak_busbw_bytes": fastest["bytes"],
    }


def build_plan(
    artifact_dir: Path,
    mode: str,
    world: int,
    per_device_gb: float,
    overhead_gb: Optional[float],
    pp_partition: str,
) -> Dict[str, Any]:
    collected = collect_artifact_tensors(artifact_dir)
    classified = classify_entries(collected["entries"])
    layer_bytes = layer_byte_table(classified)

    embed_b = bucket_bytes(classified, BUCKET_EMBED)
    head_b = bucket_bytes(classified, BUCKET_HEAD)
    norm_b = bucket_bytes(classified, BUCKET_NORM)
    sidecar_b = bucket_bytes(classified, BUCKET_SIDECAR)
    body_b = sum(layer_bytes)
    total_b = body_b + embed_b + head_b + norm_b + sidecar_b

    first_extra = embed_b
    last_extra = head_b + norm_b + sidecar_b

    budget_bytes = per_device_gb * GIB
    overhead_bytes = None if overhead_gb is None else overhead_gb * GIB

    plan: Dict[str, Any] = {
        "schema": SCHEMA,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "argv": sys.argv,
        "artifact": str(artifact_dir),
        "unit_note": GB_UNIT_NOTE,
        "parallelism": {"mode": mode, "world_size": world},
        "per_device_gb": per_device_gb,
        "per_device_gb_label": LABEL_PER_DEVICE_DEFAULT,
        "overhead_gb_per_rank": overhead_gb,
        "inventory": {
            "label": LABEL_EXACT,
            "index_source": collected["source"],
            "index": collected["index"],
            "tensor_count": len(classified),
            "num_layers": len(layer_bytes),
            "body_bytes": body_b,
            "embed_bytes": embed_b,
            "head_bytes": head_b,
            "final_norm_bytes": norm_b,
            "sidecar_bytes": sidecar_b,
            "total_classified_bytes": total_b,
            "layer_bytes": layer_bytes,
        },
        "endcap_line_items": [
            {"name": e["name"], "bucket": e["bucket"], "bytes": e["bytes"],
             "rule": e["rule"],
             "label": LABEL_ASSUMED_PLACEMENT if e["assumed_placement"] else LABEL_EXACT}
            for e in sorted(classified, key=lambda x: (-x["bytes"], x["name"]))
            if e["bucket"] in (BUCKET_EMBED, BUCKET_HEAD, BUCKET_NORM)
        ],
        "sidecar_line_items": [
            {"name": e["name"], "bytes": e["bytes"], "label": LABEL_EXACT}
            for e in sorted(classified, key=lambda x: (-x["bytes"], x["name"]))
            if e["bucket"] == BUCKET_SIDECAR
        ],
        "reported_not_classified": unclassified_payload_files(artifact_dir),
        "link_facts": _link_facts(LINK_JSON_DEFAULT),
    }

    if mode == "pp":
        even = vllm_even_split(len(layer_bytes), world)
        opt = optimal_split(layer_bytes, world, first_extra, last_extra)
        splits = {}
        for name, counts, label in (("even", even, LABEL_DERIVED_VLLM),
                                    ("optimal", opt, LABEL_EXACT_DP)):
            ranks = rank_weight_bytes(counts, layer_bytes, first_extra, last_extra)
            splits[name] = {
                "counts": list(counts),
                "ranges": [list(r) for r in split_to_ranges(counts)],
                "rank_weight_bytes": ranks,
                "max_rank_bytes": max(ranks),
                "imbalance_bytes": max(ranks) - min(ranks),
                "label": label,
                "vllm_pp_layer_partition": ",".join(str(c) for c in counts),
            }
        plan["pp"] = {
            "splits": splits,
            "selected": pp_partition,
            "endcaps": {"first_stage_bytes": first_extra,
                        "last_stage_bytes": last_extra},
        }
        sel = splits[pp_partition]
        rows = []
        for r, nbytes in enumerate(sel["rank_weight_bytes"]):
            lo, hi = sel["ranges"][r]
            row = {
                "rank": r,
                "layers": f"{lo}..{hi - 1}",
                "layer_count": hi - lo,
                "weight_bytes": nbytes,
                "overhead_bytes": overhead_bytes,
                "total_bytes": None if overhead_bytes is None else nbytes + overhead_bytes,
                "headroom_bytes": None if overhead_bytes is None
                                  else budget_bytes - nbytes - overhead_bytes,
                "headroom_label": UNKNOWN_OVERHEAD if overhead_bytes is None else LABEL_EXACT,
            }
            rows.append(row)
        plan["residency_rows"] = rows
        if overhead_bytes is None:
            plan["max_feasible"] = {"label": UNKNOWN_OVERHEAD, "feasible": None}
        else:
            mf = max_feasible_artifact_bytes(
                sel["counts"], layer_bytes, first_extra, last_extra,
                budget_bytes, overhead_bytes)
            mf["label"] = LABEL_ASSUMED_LINEAR
            mf["current_artifact_bytes"] = total_b
            plan["max_feasible"] = mf
    else:
        tp = tp_plan(classified, world)
        plan["tp"] = tp
        per_rank = tp["assumed_per_rank_bytes"]
        rows = []
        for r in range(world):
            rows.append({
                "rank": r,
                "weight_bytes": per_rank,
                "overhead_bytes": overhead_bytes,
                "total_bytes": None if overhead_bytes is None else per_rank + overhead_bytes,
                "headroom_bytes": None if overhead_bytes is None
                                  else budget_bytes - per_rank - overhead_bytes,
                "headroom_label": UNKNOWN_OVERHEAD if overhead_bytes is None
                                  else LABEL_ASSUMED_TP,
                "label": LABEL_ASSUMED_TP,
            })
        plan["residency_rows"] = rows
        plan["max_feasible"] = {
            "label": "NOT-COMPUTED-FOR-TP",
            "reason": ("a max-feasible size needs a shard policy this tool has no "
                       "attested source for; run --parallelism pp:N for that number"),
        }
    return plan


TP_CAVEAT = [
    "Every TP row above is ASSUMED.  The per-rank number models the body as",
    "fully sharded across N ranks with norms, biases, embeddings and the head",
    "replicated in full -- classified BY NAME, not by any shard policy this",
    "tool can read.  Removing the ASSUMED label needs an attested source for",
    "the serving runtime's parallel-Linear registration (which Linears are",
    "Column/RowParallel, which stay replicated, and how quantized scale",
    "tensors shard alongside their weights).  Until then, do not plan a",
    "deployment on these numbers.",
]

PP_ASSUMPTIONS = [
    "(1) body bytes scale linearly with artifact size;",
    "(2) every rank keeps the body SHARE it holds under the selected split at",
    "    today's size (shares frozen -- the split is NOT re-solved as the",
    "    artifact grows), and the endcaps (embed / head / final norm /",
    "    sidecar) stay at today's fixed size;",
    "(3) 'artifact size' means total CLASSIFIED TENSOR bytes: it excludes the",
    "    REPORTED-NOT-CLASSIFIED files and every runtime allocation (KV cache,",
    "    activations, CUDA context, graph pools) -- those live in --overhead-gb-per-rank.",
]


def render(plan: Dict[str, Any]) -> str:
    L: List[str] = []
    inv = plan["inventory"]
    mode = plan["parallelism"]["mode"]
    world = plan["parallelism"]["world_size"]
    budget_gb = plan["per_device_gb"]

    L.append("=" * 78)
    L.append("TP2 CLUSTER BUDGET PLAN -- per-rank residency and max feasible artifact")
    L.append(f"artifact     : {plan['artifact']}")
    L.append(f"parallelism  : {mode}:{world}")
    L.append(f"per-device   : {budget_gb:g} GiB  [{plan['per_device_gb_label']}]")
    ovh = plan["overhead_gb_per_rank"]
    L.append(f"overhead/rank: " + (f"{ovh:g} GiB  [ASSUMED / operator-supplied]"
                                   if ovh is not None
                                   else f"{UNKNOWN_OVERHEAD} (pass --overhead-gb-per-rank)"))
    L.append(f"units        : {plan['unit_note']}")
    L.append("=" * 78)

    L.append("")
    L.append(f"TENSOR INVENTORY  [{inv['label']}]")
    L.append(f"  source              : {inv['index_source']}")
    if inv["index"]:
        L.append(f"  shards              : {len(inv['index']['shards'])} "
                 f"(index total_size {inv['index']['declared_total_size']})")
    L.append(f"  tensors             : {inv['tensor_count']}")
    L.append(f"  decoder layers      : {inv['num_layers']} (ids 0..{inv['num_layers'] - 1})")
    L.append(f"  body (layers)       : {gib(inv['body_bytes']):>10} GiB  "
             f"({inv['body_bytes']:,} B)")
    L.append(f"  embed  -> stage 0   : {gib(inv['embed_bytes']):>10} GiB  "
             f"({inv['embed_bytes']:,} B)")
    L.append(f"  final norm -> last  : {gib(inv['final_norm_bytes']):>10} GiB  "
             f"({inv['final_norm_bytes']:,} B)")
    L.append(f"  head   -> last      : {gib(inv['head_bytes']):>10} GiB  "
             f"({inv['head_bytes']:,} B)")
    L.append(f"  sidecar-> last      : {gib(inv['sidecar_bytes']):>10} GiB  "
             f"({inv['sidecar_bytes']:,} B)")
    L.append(f"  TOTAL classified    : {gib(inv['total_classified_bytes']):>10} GiB  "
             f"({inv['total_classified_bytes']:,} B)")

    L.append("")
    L.append("ENDCAP LINE ITEMS (embed / final norm / head)")
    if not plan["endcap_line_items"]:
        L.append("  (none)")
    for item in plan["endcap_line_items"]:
        tag = " <-- ASSUMED-PLACEMENT" if item["label"] == LABEL_ASSUMED_PLACEMENT else ""
        L.append(f"  {item['bucket']:<10} {gib(item['bytes']):>9} GiB  {item['name']}"
                 f"   [{item['rule']}]{tag}")

    L.append("")
    L.append("SIDECAR LINE ITEMS (mtp / nextn / draft -- resident on the LAST stage)")
    if not plan["sidecar_line_items"]:
        L.append("  (none: this artifact ships no MTP/nextn/draft tensors)")
    for item in plan["sidecar_line_items"]:
        L.append(f"  {gib(item['bytes']):>9} GiB  {item['name']}")

    L.append("")
    L.append("REPORTED-NOT-CLASSIFIED (non-safetensors files; EXCLUDED from every "
             "residency number above and below)")
    if not plan["reported_not_classified"]:
        L.append("  (none)")
    for row in plan["reported_not_classified"]:
        L.append(f"  {row['bytes']:>14,} B  {row['file']}  [{row['kind']}]")

    if mode == "pp":
        pp = plan["pp"]
        L.append("")
        L.append("PP LAYER SPLITS")
        for name in ("even", "optimal"):
            s = pp["splits"][name]
            marker = " *SELECTED*" if name == pp["selected"] else ""
            L.append(f"  {name:<8} counts={s['vllm_pp_layer_partition']:<24} "
                     f"max-rank {gib(s['max_rank_bytes'])} GiB  "
                     f"imbalance {gib(s['imbalance_bytes'])} GiB{marker}")
            L.append(f"           [{s['label']}]")
        opt = pp["splits"]["optimal"]
        L.append(f"  VLLM_PP_LAYER_PARTITION={opt['vllm_pp_layer_partition']}"
                 f"   (realizes the optimal split)")
        L.append(f"  endcaps: stage-0 +{gib(pp['endcaps']['first_stage_bytes'])} GiB, "
                 f"last-stage +{gib(pp['endcaps']['last_stage_bytes'])} GiB")
        L.append("  NOTE: the 'even' split is DERIVED by transcribing vLLM's")
        L.append("  get_pp_indices (remainder to partitions[-2], [-3], ... so the last")
        L.append("  rank, which carries the endcaps, is skipped).  It is read from a")
        L.append("  vLLM source tree on this box, NOT attested from the pinned serving")
        L.append("  image -- re-read get_pp_indices there before trusting it in prod.")

    L.append("")
    L.append(f"PER-RANK RESIDENCY ({mode}:{world}"
             + (f", split={plan['pp']['selected']})" if mode == "pp"
                else ")  -- EVERY ROW IS ASSUMED, see TP CAVEAT"))
    hdr = (f"  {'rank':>4} {'layers':>9} {'weight GiB':>11} {'overhead GiB':>13} "
           f"{'total GiB':>10} {'headroom GiB':>26}")
    L.append(hdr)
    L.append("  " + "-" * (len(hdr) - 2))
    for row in plan["residency_rows"]:
        layers = row.get("layers", "-")
        ovh_s = "-" if row["overhead_bytes"] is None else gib(row["overhead_bytes"])
        tot_s = "-" if row["total_bytes"] is None else gib(row["total_bytes"])
        if row["headroom_bytes"] is None:
            head_s = UNKNOWN_OVERHEAD
        else:
            head_s = gib(row["headroom_bytes"])
            if row["headroom_bytes"] < 0:
                head_s += " OVER-BUDGET"
        suffix = "  ASSUMED" if mode == "tp" else ""
        L.append(f"  {row['rank']:>4} {layers:>9} {gib(row['weight_bytes']):>11} "
                 f"{ovh_s:>13} {tot_s:>10} {head_s:>26}{suffix}")
    if mode == "tp":
        tp = plan["tp"]
        L.append("")
        L.append(f"TP DECOMPOSITION  [{tp['label']}]")
        L.append(f"  ASSUMED sharded (body, split /{tp['tp_size']}): "
                 f"{gib(tp['assumed_sharded_bytes'])} GiB total -> "
                 f"{gib(tp['assumed_sharded_bytes'] / tp['tp_size'])} GiB/rank")
        L.append(f"  ASSUMED replicated (norm/bias/embed/head, full on each rank): "
                 f"{gib(tp['assumed_replicated_bytes'])} GiB/rank")
        L.append("")
        L.append("TP CAVEAT")
        for line in TP_CAVEAT:
            L.append("  " + line)

    L.append("")
    mf = plan["max_feasible"]
    if mode != "pp":
        L.append(f"MAX-FEASIBLE-ARTIFACT: {mf['label']}")
        L.append(f"  {mf['reason']}")
    elif mf.get("label") == UNKNOWN_OVERHEAD:
        L.append(f"MAX-FEASIBLE-ARTIFACT: {UNKNOWN_OVERHEAD}")
        L.append("  A max-feasible size is meaningless without the per-rank runtime")
        L.append("  overhead (KV cache + activations + CUDA context + graph pools).")
        L.append("  Pass --overhead-gb-per-rank to get the number.")
    elif not mf["feasible"]:
        L.append(f"MAX-FEASIBLE-ARTIFACT: INFEASIBLE  [{mf['label']}]")
        L.append(f"  {mf['reason']}")
    else:
        L.append(f"MAX-FEASIBLE-ARTIFACT  [{mf['label']}]")
        L.append(f"  body scale factor s_max : {mf['s_max']:.4f}x "
                 f"(binding rank {mf['binding_rank']})")
        L.append(f"  max body                : {gib(mf['max_body_bytes'])} GiB")
        L.append(f"  fixed endcaps           : {gib(mf['fixed_endcap_bytes'])} GiB")
        L.append(f"  MAX ARTIFACT            : {gib(mf['max_artifact_bytes'])} GiB "
                 f"({mf['max_artifact_bytes'] / mf['current_artifact_bytes']:.4f}x "
                 f"today's {gib(mf['current_artifact_bytes'])} GiB)")
        L.append("  assumptions behind that number:")
        for line in PP_ASSUMPTIONS:
            L.append("    " + line)

    if plan.get("link_facts"):
        lf = plan["link_facts"]
        L.append("")
        L.append("CROSS-BOX LINK CONTEXT (not used in any number above)")
        L.append(f"  {lf['provenance']}")
        L.append(f"    p50 {lf['small_msg_p50_us']:.1f} us at {lf['small_msg_bytes']} B; "
                 f"peak busbw {lf['peak_busbw_GBps']:.1f} GB/s at "
                 f"{lf['peak_busbw_bytes']} B")
        L.append(f"    source: {lf['source']}")
        L.append("  This planner models RESIDENCY only.  For the latency/bandwidth")
        L.append("  arithmetic of TP=2 decode, use tools/tp_decode_feasibility.py.")
    L.append("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-rank residency and max-feasible-artifact planner for the "
                    "two-DGX-Spark cluster.")
    ap.add_argument("--artifact", required=True, type=Path,
                    help="artifact directory (safetensors + index)")
    ap.add_argument("--parallelism", required=True,
                    help="pp:N or tp:N (N >= 2)")
    ap.add_argument("--per-device-gb", type=float, default=PER_DEVICE_GB_DEFAULT,
                    help=f"usable GiB per rank (default {PER_DEVICE_GB_DEFAULT:g}, ASSUMED)")
    ap.add_argument("--overhead-gb-per-rank", type=float, default=None,
                    help="runtime overhead GiB per rank; omitted -> "
                         f"{UNKNOWN_OVERHEAD} instead of an invented number")
    ap.add_argument("--pp-partition", choices=("even", "optimal"), default="optimal",
                    help="which split drives the residency table and the "
                         "max-feasible inversion (both are always reported)")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="write the full plan as JSON here (never under /tmp)")
    args = ap.parse_args(argv)

    mode, world = parse_parallelism(args.parallelism)
    artifact = args.artifact.expanduser().resolve()
    if not artifact.is_dir():
        print(f"error: --artifact {artifact} is not a directory", file=sys.stderr)
        return 2
    try:
        plan = build_plan(artifact, mode, world, args.per_device_gb,
                          args.overhead_gb_per_rank, args.pp_partition)
    except (ClassificationError, HeaderError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(render(plan))
    if args.json_out is not None:
        out = check_out_path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
