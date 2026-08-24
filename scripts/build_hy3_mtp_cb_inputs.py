#!/usr/bin/env python3
"""Build the MTP-sidecar layer_config + uniform col-weights for the Hy3 CB draft.

The Hy3 body ships CB-quantised (``prod-hy3-nvfp4cb-2p9``) but its bf16 MTP
draft (``model.layers.80.*``, 7.5 GB) OOMs next to ~102 GiB of weights on the
128 GB Spark. Robert's rule is "always include MTP when it's available", so the
MTP module is CB-quantised too. A draft's weights can NEVER change outputs
(spec-decode is exact via rejection sampling) — they only move the acceptance
rate — so the rung is a **throughput** choice, not a quality one.

Two rung-selection policies (``--rung-select``):

  * ``modal`` (default; reproduces the shipped sidecar byte-for-byte):
      - routed experts -> the body's **modal fp4 expert rung**;
      - shared expert + attention -> **FP8_CB_K32** (the sensitive-tail default).
  * ``auto`` (the canon selector, ``docs/design/mtp_rung_selection.md``): build the
    per-role CB menu, fit served acceptance to ``a(b)=a_inf−β·sqrt(E(b))``,
    and pick the throughput-optimal rung with ``mtp_rung_selection.select_rung``.
    Emits ``mtp_rung_selection.json`` (the full selection provenance) next to
    the layer_config. Exact menu bytes are derived from the body assignment's
    persisted CB serialization context; an unstamped body assignment is rejected.
    Default stays ``modal`` so the shipped artifact is stable.

Everything the MTP layer_config does NOT name (``enorm``/``hnorm``/``eh_proj``/
``final_layernorm``, the block's norms, ``q_norm``/``k_norm``, the router gate,
``expert_bias``) stays bf16/f32 — carried through unquantised, exactly like the
body-layer conventions.

Outputs (into ``--out-dir``): ``mtp_layer_config.json`` and
``mtp_col_weights.pkl`` (both keyed by recipe qnames), plus, in ``auto`` mode,
``mtp_rung_selection.json``. col-weights are **uniform** (all-ones per input
column): no imatrix — the draft-quality caveat above.

Shapes are read from safetensors metadata (CPU-only). ``auto`` mode needs a
draft error curve ``E(b)`` per rung: either a precomputed ``--e-table`` JSON
(CPU) or ``--measure-e`` (GPU: RTN weight-MSE on the actual MTP weights). E is
NEVER fabricated — one of the two is required in ``auto`` mode.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open

from prismaquant import format_registry as fr
from prismaquant.model_profiles import detect_profile
from prismaquant.mtp_rung_selection import (
    AcceptancePoint,
    RungPoint,
    ServeConstants,
    select_rung,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    cb_payload_summary,
    cb_quantize_dequantize_for_context,
    cb_serialization_context_from_stamp,
    cb_serialization_context_stamp,
    cb_serialization_metadata_from_assignment_payload,
)

# Recipe-qname roles under model.layers.{L}. and their target CB family.
_ATTN_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")
_SHARED_LEAVES = ("gate_proj", "up_proj", "down_proj")
_EXPERT_LEAVES = ("gate_up_proj", "down_proj")

# The auto-mode draft menu (doc §3 / §5). Experts may sit on fp4 (cheap) or fp8
# (accurate) families; dense/shared (attention + shared expert) are fp8 only.
_EXPERT_FP4_KS = (14, 15, 16, 17, 18, 19, 20)      # NVFP4_CB, 2.25..3.0 bpw
_EXPERT_FP8_KS = (28, 32, 36, 40, 44)              # FP8_CB,  3.5..5.5 bpw
_DENSE_FP8_KS = (28, 32, 36, 40, 44)               # FP8_CB,  3.5..5.5 bpw

# Hy3 2026-07-20 serve constants (doc §5): eager drafter, k=1 forced.
_HY3_T_MS = 76.0
_HY3_D0_MS = 50.0
_HY3_C_MS_PER_BIT = 0.1
# Draft memory budget default (GiB). NOTE: this is the budget for the DRAFT's
# resident bytes only — the caller must net weights + profiling-peak + the 3 GiB
# margin out of the ~121 GiB usable pool before overriding it (doc §3.5).
_HY3_MEM_BUDGET_GB = 16.0


# --------------------------------------------------------------------------- #
# Source-checkpoint plumbing (metadata only; no tensor data)
# --------------------------------------------------------------------------- #
def _shard_map(model_dir: Path) -> dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(
            f"no model.safetensors[.index.json] under {model_dir}")
    with safe_open(single, framework="pt", device="cpu") as f:
        return {k: "model.safetensors" for k in f.keys()}


def _shape(model_dir: Path, shard_map: dict[str, str], name: str):
    if name not in shard_map:
        raise KeyError(f"{name}: not in source checkpoint")
    with safe_open(model_dir / shard_map[name], framework="pt",
                   device="cpu") as f:
        return tuple(f.get_slice(name).get_shape())


def _expert_ids(shard_map: dict[str, str], base: str) -> list[int]:
    """Sorted routed-expert indices present under ``{base}.mlp.experts.{e}.``."""
    rx = re.compile(re.escape(base) + r"\.mlp\.experts\.(\d+)\.")
    ids = {int(m.group(1)) for k in shard_map for m in [rx.match(k)] if m}
    if not ids:
        raise ValueError(f"no routed experts under {base}.mlp.experts.*")
    return sorted(ids)


# --------------------------------------------------------------------------- #
# Scheme resolution (clone the body's scheme; synthesise from the registry only
# for auto-mode rungs the body never used)
# --------------------------------------------------------------------------- #
def _canonical_entries(body_lc: dict) -> dict[tuple[str, int], dict]:
    """{(data_type, cb_k): a representative entry dict} from the body config —
    cloned verbatim so a chosen rung's scheme fields are byte-identical to the
    body's for that rung."""
    out: dict[tuple[str, int], dict] = {}
    for v in body_lc.values():
        dt, k = v.get("data_type"), v.get("cb_k")
        if dt in ("nvfp4_cb", "fp8_cb") and k is not None:
            out.setdefault((dt, int(k)), dict(v))
    return out


def _cb_format_name(key: tuple[str, int]) -> str:
    dt, k = key
    return f"{'NVFP4' if dt == 'nvfp4_cb' else 'FP8'}_CB_K{k}"


def _resolve_scheme(key: tuple[str, int], canon: dict,
                    allow_synthesize: bool) -> tuple[dict, str]:
    """Return (layer_config entry, source) for a (data_type, cb_k) rung.

    Prefer cloning the body's scheme (reproducible); for an auto-mode rung the
    body never used, synthesise the scheme from the format registry's
    autoround_config (equivalent scheme fields). ``modal`` never synthesises.
    """
    if key in canon:
        return dict(canon[key]), "body_clone"
    if not allow_synthesize:
        raise ValueError(
            f"body layer_config has no {key[0]} cb_k={key[1]} entry to clone "
            "the scheme from — pick an existing rung")
    spec = fr.get_format(_cb_format_name(key))
    cfg = spec.autoround_config
    entry = dict(cfg() if callable(cfg) else cfg)
    return entry, "registry_synth"


def _modal_expert_fp4_k(body_lc: dict) -> tuple[int, Counter]:
    """The modal fp4 cb_k over the body's routed-expert entries."""
    dist: Counter = Counter()
    for q, v in body_lc.items():
        if ".mlp.experts." in q and q.rsplit(".", 1)[1] in _EXPERT_LEAVES \
                and v.get("data_type") == "nvfp4_cb":
            dist[int(v["cb_k"])] += 1
    if not dist:
        raise ValueError("no nvfp4_cb routed-expert entries in the body "
                         "layer_config — cannot pick a modal fp4 expert rung")
    return dist.most_common(1)[0][0], dist


def _load_body_cb_serialization_context(
    body_layer_config: str | Path,
) -> CBSerializationContext:
    """Load the exact CB producer identity carried by the body assignment.

    Auto selection makes resident-byte and render-error claims, so it may not
    infer layout-v1/v2, scale sweep, encoder tier, or codebook sharing from the
    current process defaults. Historical unstamped assignments remain usable by
    the modal builder, which makes no exact byte claim; auto mode fails closed.
    """
    path = Path(body_layer_config)
    payload = json.loads(path.read_text())
    stamp, _identities = cb_serialization_metadata_from_assignment_payload(
        payload
    )
    if stamp is None:
        raise ValueError(
            f"{path}: auto MTP selection requires the body assignment's "
            "explicit cb_serialized_payload context (layout/scale coding, "
            "scale sweep, encoder tier, and codebook identity); refusing to "
            "price a legacy unstamped CB menu"
        )
    return cb_serialization_context_from_stamp(
        stamp,
        where=f"Hy3 MTP body assignment {path}",
    )


# --------------------------------------------------------------------------- #
# The build (shared by modal + auto). expert_key / shared_key are
# (data_type, cb_k) tuples; when None they default to the modal policy.
# --------------------------------------------------------------------------- #
def build(source_dir: Path, body_layer_config: Path, out_dir: Path,
          mtp_layer: int, expert_fp4_k: int | None,
          shared_attn_fp8_k: int, *,
          expert_key: tuple[str, int] | None = None,
          shared_key: tuple[str, int] | None = None,
          allow_synthesize: bool = False) -> dict:
    body_lc = json.loads(Path(body_layer_config).read_text())
    canon = _canonical_entries(body_lc)
    modal_k, dist = _modal_expert_fp4_k(body_lc)
    if expert_key is None:
        ek = expert_fp4_k if expert_fp4_k is not None else modal_k
        expert_key = ("nvfp4_cb", ek)
    if shared_key is None:
        shared_key = ("fp8_cb", shared_attn_fp8_k)

    profile = detect_profile(str(source_dir))
    if profile is None:
        raise RuntimeError(f"no model profile detected for {source_dir}")
    shard_map = _shard_map(Path(source_dir))
    L = mtp_layer
    base = f"model.layers.{L}"

    # (recipe qname, (data_type, cb_k), source-key-for-in_features)
    targets: list[tuple[str, tuple[str, int], str]] = []
    for leaf in _ATTN_LEAVES:
        q = f"{base}.self_attn.{leaf}"
        targets.append((q, shared_key, profile.source_tensor_name(q) + ".weight"))
    for leaf in _SHARED_LEAVES:
        q = f"{base}.mlp.shared_experts.{leaf}"
        targets.append((q, shared_key, profile.source_tensor_name(q) + ".weight"))
    targets.append((f"{base}.mlp.experts.gate_up_proj", expert_key,
                    f"{base}.mlp.experts.0.gate_proj.weight"))
    targets.append((f"{base}.mlp.experts.down_proj", expert_key,
                    f"{base}.mlp.experts.0.down_proj.weight"))

    layer_config: dict[str, dict] = {}
    col_weights: dict[str, torch.Tensor] = {}
    scheme_sources: dict[str, str] = {}
    for qname, key, src in targets:
        entry, source = _resolve_scheme(key, canon, allow_synthesize)
        layer_config[qname] = entry
        scheme_sources[qname] = source
        in_features = int(_shape(Path(source_dir), shard_map, src)[1])
        if in_features % 256 != 0:
            raise ValueError(
                f"{qname}: in_features={in_features} not a multiple of 256 "
                "(CB superblock) — cannot CB-quantise")
        col_weights[qname] = torch.ones(in_features, dtype=torch.float32)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lc_path = out_dir / "mtp_layer_config.json"
    cw_path = out_dir / "mtp_col_weights.pkl"
    lc_path.write_text(json.dumps(layer_config, indent=2, sort_keys=True))
    with open(cw_path, "wb") as fh:
        pickle.dump({k: v for k, v in col_weights.items()}, fh)

    print(f"[mtp-inputs] modal fp4 expert rung: NVFP4_CB_K{modal_k}  "
          f"(fp4 dist over experts: {dict(sorted(dist.items()))})")
    print(f"[mtp-inputs] experts   -> {_cb_format_name(expert_key)}")
    print(f"[mtp-inputs] shared+attn -> {_cb_format_name(shared_key)}")
    if "registry_synth" in scheme_sources.values():
        print("[mtp-inputs] NOTE: some schemes synthesised from the registry "
              "(rung not present in the body layer_config)")
    print(f"[mtp-inputs] {len(layer_config)} CB targets for layer {L}:")
    for q in sorted(layer_config):
        e = layer_config[q]
        fmt = (f"NVFP4_CB_K{e['cb_k']}" if e["data_type"] == "nvfp4_cb"
               else f"FP8_CB_K{e['cb_k']}")
        print(f"    {q:52s} {fmt:14s} in={col_weights[q].numel()}")
    print(f"[mtp-inputs] wrote {lc_path}")
    print(f"[mtp-inputs] wrote {cw_path}")
    return {"layer_config": str(lc_path), "col_weights": str(cw_path),
            "expert_key": list(expert_key), "shared_key": list(shared_key),
            "n_targets": len(layer_config), "scheme_sources": scheme_sources}


# --------------------------------------------------------------------------- #
# Auto mode: menu construction, E resolution, selection
# --------------------------------------------------------------------------- #
def _expert_family_for_k(k: int, *, family: str | None = None) -> str:
    from prismaquant.cb_layout import FAMILIES

    matches = [
        cb_family.prefix.removesuffix("_K").lower()
        for cb_family in FAMILIES
        if cb_family.is_producer_rung(k)
    ]
    if family is not None:
        requested = str(family).strip().lower()
        if requested not in matches:
            raise ValueError(
                f"cb_k={k} is not a producer rung of {requested!r}; "
                f"eligible families are {matches}"
            )
        return requested
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"cb_k={k} maps to no producer-eligible CB family")
    raise ValueError(
        f"cb_k={k} is ambiguous across producer CB families {matches}; "
        "pass family='nvfp4_cb' or family='fp8_cb'"
    )


def _expert_shape_bytes(source_dir: Path, shard_map: dict, base: str):
    """(num_experts, [gate0, up0, down0] 2-D shapes) for the routed experts."""
    ids = _expert_ids(shard_map, base)
    shapes = [
        _shape(source_dir, shard_map, f"{base}.mlp.experts.{ids[0]}.gate_proj.weight"),
        _shape(source_dir, shard_map, f"{base}.mlp.experts.{ids[0]}.up_proj.weight"),
        _shape(source_dir, shard_map, f"{base}.mlp.experts.{ids[0]}.down_proj.weight"),
    ]
    return len(ids), shapes


def _num_params(shape) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def _auto_target_shapes(source_dir: Path, shard_map: dict, profile, base: str):
    """Return the concrete packed-expert and dense MTP serialization shapes."""
    num_experts, (g0, u0, d0) = _expert_shape_bytes(
        source_dir, shard_map, base
    )
    dense_shapes: dict[str, tuple[int, ...]] = {}
    for leaf in _ATTN_LEAVES:
        qname = f"{base}.self_attn.{leaf}"
        dense_shapes[qname] = _shape(
            source_dir,
            shard_map,
            profile.source_tensor_name(qname) + ".weight",
        )
    for leaf in _SHARED_LEAVES:
        qname = f"{base}.mlp.shared_experts.{leaf}"
        dense_shapes[qname] = _shape(
            source_dir,
            shard_map,
            profile.source_tensor_name(qname) + ".weight",
        )
    if int(g0[1]) != int(u0[1]):
        raise ValueError(
            f"{base}: gate/up expert input widths differ: {g0[1]} != {u0[1]}"
        )
    expert_shapes = {
        f"{base}.mlp.experts.gate_up_proj": (
            num_experts,
            int(g0[0]) + int(u0[0]),
            int(g0[1]),
        ),
        f"{base}.mlp.experts.down_proj": (
            num_experts,
            int(d0[0]),
            int(d0[1]),
        ),
    }
    return num_experts, dense_shapes, expert_shapes


def _exact_scope_rate(
    shapes: dict[str, tuple[int, ...]],
    format_name: str,
    context: CBSerializationContext,
) -> tuple[dict, float]:
    assignment = {qname: format_name for qname in shapes}
    payload = cb_assignment_payload_breakdown(
        assignment,
        shapes,
        context=context,
    )
    params = sum(_num_params(shape) for shape in shapes.values())
    return payload, 8.0 * int(payload["total_bytes"]) / max(params, 1)


def _pair_dense_key_exact(
    expert_key: tuple[str, int],
    dense_shapes: dict[str, tuple[int, ...]],
    expert_shapes: dict[str, tuple[int, ...]],
    context: CBSerializationContext,
) -> tuple[str, int]:
    """Match dense/shared FP8 to the expert tier using exact scope rates.

    The comparison includes layout-v1/v2, FP8 row scales, and one deduplicated
    sidecar set within each role scope. The final full-draft payload is priced
    again below so a sidecar shared by expert and dense scopes is charged once.
    """
    expert_format = _cb_format_name(expert_key)
    _payload, expert_bpp = _exact_scope_rate(
        expert_shapes,
        expert_format,
        context,
    )
    candidates: list[tuple[float, int]] = []
    for dense_k in _DENSE_FP8_KS:
        _dense_payload, dense_bpp = _exact_scope_rate(
            dense_shapes,
            _cb_format_name(("fp8_cb", dense_k)),
            context,
        )
        candidates.append((abs(dense_bpp - expert_bpp), dense_k))
    # Equal distance prefers the higher-fidelity dense rung.
    dense_k = min(candidates, key=lambda item: (item[0], -item[1]))[1]
    return "fp8_cb", dense_k


def _build_auto_menu_from_shapes(
    *,
    num_experts: int,
    dense_shapes: dict[str, tuple[int, ...]],
    expert_shapes: dict[str, tuple[int, ...]],
    e_table: dict[str, float],
    cb_context: CBSerializationContext,
) -> tuple[list[RungPoint], dict]:
    """Build an exact serialized-payload menu from concrete target shapes."""
    if not isinstance(cb_context, CBSerializationContext):
        raise ValueError(
            "Hy3 MTP auto menu requires an explicit CBSerializationContext; "
            "refusing FormatSpec/default byte estimates"
        )
    expert_ladder = [("nvfp4_cb", k) for k in _EXPERT_FP4_KS] + [
        ("fp8_cb", k) for k in _EXPERT_FP8_KS
    ]
    all_shapes = {**dense_shapes, **expert_shapes}
    total_params = sum(_num_params(shape) for shape in all_shapes.values())

    menu: list[RungPoint] = []
    pairing: dict[str, dict] = {}
    used_formats: set[str] = set()
    for expert_key in expert_ladder:
        _expert_dt, expert_k = expert_key
        name = f"K{expert_k}"
        dense_key = _pair_dense_key_exact(
            expert_key,
            dense_shapes,
            expert_shapes,
            cb_context,
        )
        expert_format = _cb_format_name(expert_key)
        dense_format = _cb_format_name(dense_key)
        assignment = {
            **{qname: expert_format for qname in expert_shapes},
            **{qname: dense_format for qname in dense_shapes},
        }
        payload = cb_assignment_payload_breakdown(
            assignment,
            all_shapes,
            context=cb_context,
        )
        total_bytes = int(payload["total_bytes"])
        bits = 8.0 * total_bytes / max(total_params, 1)
        if name not in e_table:
            raise KeyError(
                f"auto menu rung {name} ({expert_format}) has no E in the "
                "e-table — measure it (--measure-e) or add it to --e-table "
                "(never fabricate E)"
            )
        menu.append(
            RungPoint(
                name=name,
                bits=bits,
                resident_bytes=total_bytes,
                E=float(e_table[name]),
            )
        )
        used_formats.update((expert_format, dense_format))
        pairing[name] = {
            "expert_key": list(expert_key),
            "dense_key": list(dense_key),
            "expert_format": expert_format,
            "dense_format": dense_format,
            "bits": bits,
            "resident_bytes": total_bytes,
            "byte_scope": (
                "cb_tensor_data_plus_deduplicated_fp16_codebook_sidecars"
            ),
            "serialized_payload": cb_payload_summary(payload),
        }
    return menu, {
        "num_experts": int(num_experts),
        "total_params": total_params,
        "byte_scope": (
            "cb_tensor_data_plus_deduplicated_fp16_codebook_sidecars"
        ),
        "cb_serialized_payload": cb_serialization_context_stamp(
            cb_context,
            formats=sorted(used_formats),
        ),
        "pairing": pairing,
    }


def _build_auto_menu(source_dir: Path, shard_map: dict, profile, base: str,
                     e_table: dict[str, float], *,
                     cb_context: CBSerializationContext
                     ) -> tuple[list[RungPoint], dict]:
    """Build the per-rung menu (bits, resident bytes, E) for the draft.

    Each menu rung is a full-draft encoding named by its EXPERT rung short name
    (``K14``..``K20`` fp4, ``K28``..``K44`` fp8); dense/shared ride along at the
    nearest exact-rate fp8 tier. Bits and resident bytes come from the
    authoritative serialized-payload API over the real packed target shapes;
    E comes from ``e_table`` (measured or precomputed).
    """
    num_experts, dense_shapes, expert_shapes = _auto_target_shapes(
        source_dir,
        shard_map,
        profile,
        base,
    )
    return _build_auto_menu_from_shapes(
        num_experts=num_experts,
        dense_shapes=dense_shapes,
        expert_shapes=expert_shapes,
        e_table=e_table,
        cb_context=cb_context,
    )


def _normalize_rung_name(s: str) -> str:
    """Map an accept-point / e-table key to a menu name: 'NVFP4_CB_K18'|'K18'|'18'
    -> 'K18'."""
    m = re.search(r"K?(\d+)$", str(s).strip().upper())
    if not m:
        raise ValueError(f"cannot parse rung name from {s!r}")
    return f"K{int(m.group(1))}"


def _measure_e_table(source_dir: Path, shard_map: dict, profile, base: str, *,
                     cb_context: CBSerializationContext
                     ) -> dict[str, float]:
    """Uniform-h Σ per-Linear RTN weight-MSE per menu rung, on the real MTP
    weights (doc §2/§3). GPU-only: loads and CB-quantises every draft Linear at
    each rung's per-role format. Fails fast without CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("--measure-e requires CUDA (RTN weight-MSE on the "
                           "MTP weights); provide --e-table instead on CPU")
    dev = torch.device("cuda")
    ids = _expert_ids(shard_map, base)
    if cb_context.codebook_source != "lattice":
        raise ValueError(
            "--measure-e cannot render a learned CB context without the exact "
            "materialized codebook values; provide a value-bearing renderer or "
            "use a precomputed --e-table bound to that context"
        )
    _num_experts, dense_shapes, expert_shapes = _auto_target_shapes(
        source_dir,
        shard_map,
        profile,
        base,
    )

    def _load(name: str) -> torch.Tensor:
        with safe_open(source_dir / shard_map[name], framework="pt",
                       device="cpu") as f:
            return f.get_tensor(name).to(dev, torch.float32)

    def _mse(w: torch.Tensor, fmt_name: str) -> float:
        spec = fr.get_format(fmt_name)
        # The MTP policy deliberately uses uniform column importance, but the
        # production CB renderer is still weighted VQ. Bind the all-ones values
        # explicitly along with layout, scale sweep and encoder tier.
        col_weights = torch.ones(
            int(w.shape[-1]),
            dtype=torch.float32,
            device=w.device,
        )
        wq = cb_quantize_dequantize_for_context(
            spec,
            w.clone(),
            context=cb_context,
            col_weights=col_weights,
        )
        return float(((w - wq.to(w.dtype)) ** 2).mean().item())

    dense_names = (
        [profile.source_tensor_name(f"{base}.self_attn.{l}") + ".weight"
         for l in _ATTN_LEAVES]
        + [profile.source_tensor_name(f"{base}.mlp.shared_experts.{l}") + ".weight"
           for l in _SHARED_LEAVES])
    dense_ws = [_load(n) for n in dense_names]

    expert_ladder = [("nvfp4_cb", k) for k in _EXPERT_FP4_KS] + \
                    [("fp8_cb", k) for k in _EXPERT_FP8_KS]
    e_table: dict[str, float] = {}
    for ekey in expert_ladder:
        name = f"K{ekey[1]}"
        efmt = _cb_format_name(ekey)
        dkey = _pair_dense_key_exact(
            ekey,
            dense_shapes,
            expert_shapes,
            cb_context,
        )
        dfmt = _cb_format_name(dkey)
        E = sum(_mse(w, dfmt) for w in dense_ws)
        for e in ids:
            for leaf in ("gate_proj", "up_proj", "down_proj"):
                E += _mse(_load(f"{base}.mlp.experts.{e}.{leaf}.weight"), efmt)
        e_table[name] = E
        print(f"[mtp-inputs] measured E[{name}] ({efmt}/{dfmt}) = {E:.6g}")
    return e_table


def run_auto(args) -> dict:
    source_dir = Path(args.source_dir)
    profile = detect_profile(str(source_dir))
    if profile is None:
        raise RuntimeError(f"no model profile detected for {source_dir}")
    shard_map = _shard_map(source_dir)
    base = f"model.layers.{args.mtp_layer}"
    cb_context = _load_body_cb_serialization_context(
        args.body_layer_config
    )

    # E(b): measured on GPU or a precomputed table — never fabricated.
    if args.measure_e and args.e_table:
        raise ValueError("pass only one of --measure-e / --e-table")
    if args.measure_e:
        e_table = _measure_e_table(
            source_dir,
            shard_map,
            profile,
            base,
            cb_context=cb_context,
        )
        e_source = "measured_rtn_weight_mse"
    elif args.e_table:
        raw = json.loads(Path(args.e_table).read_text())
        e_table = {_normalize_rung_name(k): float(v) for k, v in raw.items()}
        e_source = f"e_table:{args.e_table}"
    else:
        raise ValueError("auto mode needs an error curve: --measure-e (GPU) or "
                         "--e-table JSON (never fabricate E)")

    menu, menu_meta = _build_auto_menu(
        source_dir,
        shard_map,
        profile,
        base,
        e_table,
        cb_context=cb_context,
    )

    accept_points = []
    for spec in args.accept_point or []:
        if "=" not in spec:
            raise ValueError(f"--accept-point wants RUNG=VALUE, got {spec!r}")
        rung, val = spec.split("=", 1)
        accept_points.append(AcceptancePoint(float(val),
                                             rung_name=_normalize_rung_name(rung)))

    constants = ServeConstants(t_ms=args.t_ms, d0_ms=args.d0_ms,
                               c_ms_per_bit=args.c_ms_per_bit)
    mem_budget_bytes = int(args.mem_budget_gb * (1024 ** 3))
    result = select_rung(menu, constants, accept_points, mem_budget_bytes,
                         k=args.k, h_source="uniform")

    chosen = menu_meta["pairing"][result.rung.name]
    expert_key = tuple(chosen["expert_key"])
    shared_key = tuple(chosen["dense_key"])
    print(f"[mtp-inputs] AUTO selected rung {result.rung.name} "
          f"(regime={result.regime}): experts {chosen['expert_format']}, "
          f"dense {chosen['dense_format']}")

    out = build(source_dir, Path(args.body_layer_config), Path(args.out_dir),
                args.mtp_layer, args.expert_fp4_k, args.shared_attn_fp8_k,
                expert_key=expert_key, shared_key=shared_key,
                allow_synthesize=True)

    provenance = dict(result.provenance)
    provenance["driver"] = {
        "source_dir": str(source_dir), "mtp_layer": args.mtp_layer,
        "e_source": e_source, "num_experts": menu_meta["num_experts"],
        "total_quant_params": menu_meta["total_params"],
        "byte_scope": menu_meta["byte_scope"],
        "cb_serialized_payload": menu_meta["cb_serialized_payload"],
        "pairing": menu_meta["pairing"],
        "selected_expert_format": chosen["expert_format"],
        "selected_dense_format": chosen["dense_format"],
    }
    prov_path = Path(args.out_dir) / "mtp_rung_selection.json"
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    print(f"[mtp-inputs] wrote {prov_path}")
    out["rung_selection"] = str(prov_path)
    out["selected_rung"] = result.rung.name
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", required=True,
                    help="HF bf16 source (has model.layers.{L}.* MTP tensors)")
    ap.add_argument("--body-layer-config", required=True,
                    help="the body artifact's layer_config.json (rung source)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mtp-layer", type=int, default=80,
                    help="the MTP sidecar layer index (num_hidden_layers)")
    ap.add_argument("--rung-select", choices=("modal", "auto"), default="modal",
                    help="modal = mirror the body's dominant rung (reproduces "
                         "the shipped sidecar); auto = the canon throughput "
                         "selector (docs/design/mtp_rung_selection.md)")
    # modal knobs (also the auto defaults for the modal-equivalent build call)
    ap.add_argument("--expert-fp4-k", type=int, default=None,
                    help="modal: override the routed-expert fp4 rung (default modal)")
    ap.add_argument("--shared-attn-fp8-k", type=int, default=32,
                    help="modal: the shared-expert + attention fp8 rung (default 32)")
    # auto knobs
    ap.add_argument("--accept-point", action="append", metavar="RUNG=VALUE",
                    help="auto: served acceptance measurement, e.g. K18=0.85 "
                         "(repeatable; >=2 rungs give a fitted slope)")
    ap.add_argument("--e-table", default=None,
                    help="auto: JSON {rung: E(b)} draft error curve (CPU path)")
    ap.add_argument("--measure-e", action="store_true",
                    help="auto: measure E(b) as uniform-h Σ RTN weight-MSE on "
                         "the real MTP weights (GPU-only)")
    ap.add_argument("--k", type=int, default=1,
                    help="auto: speculative tokens per cycle (Hy3 forces k=1)")
    ap.add_argument("--t-ms", type=float, default=_HY3_T_MS,
                    help="auto: target verify-step time (ms)")
    ap.add_argument("--d0-ms", type=float, default=_HY3_D0_MS,
                    help="auto: rung-independent drafter overhead (ms)")
    ap.add_argument("--c-ms-per-bit", type=float, default=_HY3_C_MS_PER_BIT,
                    help="auto: drafter time per bit/weight (ms/bit)")
    ap.add_argument("--mem-budget-gb", type=float, default=_HY3_MEM_BUDGET_GB,
                    help="auto: DRAFT resident-byte budget (GiB); caller nets "
                         "weights + profiling-peak + 3 GiB margin out first")
    args = ap.parse_args(argv)

    if args.rung_select == "auto":
        run_auto(args)
    else:
        build(Path(args.source_dir), Path(args.body_layer_config),
              Path(args.out_dir), args.mtp_layer, args.expert_fp4_k,
              args.shared_attn_fp8_k)


if __name__ == "__main__":
    main()
