"""Build a Sensitivity Card from a ``probe.pkl``, and inspect/publish it.

Two entry points:

* :func:`card_from_probe` -- convert an existing probe to a card. Probes taken
  before marginal emission existed convert fine; they simply produce a
  scalar-only card, which is *degraded, not broken*: it reproduces exactly
  today's allocator behaviour and nothing more. That property is what makes the
  card safe to adopt without re-probing anything.

* ``python -m prismaquant.sensitivity_card_build`` -- a CLI to build, inspect
  and size a card.

The card deliberately carries no serving policy. Fused-sibling and packed-expert
*identity* is model structure and travels with the card; the rule that siblings
must share a format is the consumer's platform contract and is derived
downstream from whichever profile the author names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from .sensitivity_card import (
    CardProvenance,
    RenderBasis,
    SensitivityCard,
    SensitivityUnit,
    UnitTopology,
)

VECTOR_KEYS = ("fisher_row", "fisher_col", "g_sq_sum", "act_sq_sum", "act_absmax")

# The packed-expert A-side arrays. Kept apart from VECTOR_KEYS because they are
# [E, *] matrices, and `_as_vector` deliberately rejects anything that is not a
# 1-D per-Linear vector.
EXPERT_KEYS = ("expert_g_sq_sum", "expert_act_sq_sum", "expert_act_absmax",
               "expert_tokens")

# Role is read off the module name. This is descriptive metadata for consumers
# that price per-role; it never gates a format, so an unrecognised name simply
# yields None rather than an error.
_ROLE_PATTERNS = (
    (re.compile(r"\bq_proj\b"), "q"),
    (re.compile(r"\bk_proj\b"), "k"),
    (re.compile(r"\bv_proj\b"), "v"),
    (re.compile(r"\b(o_proj|out_proj)\b"), "o"),
    (re.compile(r"\b(gate_proj|w1)\b"), "gate"),
    (re.compile(r"\b(up_proj|w3)\b"), "up"),
    (re.compile(r"\b(down_proj|w2)\b"), "down"),
    (re.compile(r"\bw13\b"), "gate_up"),
    (re.compile(r"\bqkv\b"), "qkv"),
)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


def _role_of(name: str) -> str | None:
    for pat, role in _ROLE_PATTERNS:
        if pat.search(name):
            return role
    return None


def _layer_of(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _fused_group_of(name: str, role: str | None, layer: int | None) -> str | None:
    """Sibling identity: q/k/v of one block, gate/up of one MLP.

    Derived from the module path rather than from a profile so that a card can
    be built without loading the model. A consumer that needs authoritative
    grouping re-derives it from its own profile; this is a portable hint.
    """
    if layer is None or role is None:
        return None
    if role in ("q", "k", "v"):
        return f"L{layer}.attn_qkv"
    if role in ("gate", "up"):
        prefix = name.rsplit(".", 1)[0]
        return f"{prefix}.gate_up"
    return None


def _packed_group_of(name: str, layer: int | None) -> tuple[str | None, int | None]:
    m = _EXPERT_RE.search(name)
    if m is None or layer is None:
        return None, None
    expert_id = int(m.group(1))
    leaf = name.rsplit(".", 1)[-1]
    return f"L{layer}.experts.{leaf}", expert_id


def _as_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    return arr if arr.ndim == 1 and arr.size else None


def _as_expert_array(value: Any, ndim: int) -> np.ndarray | None:
    """A packed-expert marginal: [E, C] for the sums, [E] for the counts."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    return arr if arr.ndim == ndim and arr.size else None


def _git_commit(repo: str | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _calib_hash(meta: Mapping[str, Any]) -> str:
    """Prefer the probe's own recorded hash; otherwise derive one.

    Calibration is identity, so a card must always carry *some* stable
    calibration fingerprint. A derived one is marked so it is never mistaken
    for the authoritative hash the pipeline stamps.
    """
    for key in ("calib_hash", "calibration_hash", "calib_sha256"):
        if meta.get(key):
            return str(meta[key])
    payload = json.dumps(
        {k: str(meta.get(k)) for k in sorted(
            ("dataset", "nsamples", "seqlen", "seed", "model"))},
        sort_keys=True)
    return "derived:" + hashlib.sha256(payload.encode()).hexdigest()[:56]


def card_from_probe(
    probe_path: str,
    *,
    model_id: str | None = None,
    probe_commit: str | None = None,
    render_basis: RenderBasis = RenderBasis.RTN,
    notes: str = "",
) -> SensitivityCard:
    """Convert ``probe.pkl`` into a Sensitivity Card.

    Raw (token-sum) accumulators are carried through unnormalized together with
    the global token count, because a normalized value cannot be un-normalized
    and different consumers legitimately want different normalizations.

    MoE note: every row -- dense or expert -- is normalized by the SAME global
    calibration token count. Dividing an expert row by its own routed-token
    count inverts importance weighting (the bug removed in PR #14); ``route_prob``
    is carried for diagnostics only and must not be used to rescale.
    """
    with open(probe_path, "rb") as fh:
        probe = pickle.load(fh)

    stats: Mapping[str, Mapping[str, Any]] = probe.get("stats", {})
    meta: Mapping[str, Any] = probe.get("meta", {}) or {}

    # The global token count is the normalizer for every row. Fall back to the
    # max observed per-row count, which is the dense rows' value.
    global_tokens = int(
        meta.get("global_tokens")
        or meta.get("n_tokens_total")
        or max((int(s.get("n_tokens_seen", 0) or 0) for s in stats.values()),
               default=0)
        or 1)

    units: list[SensitivityUnit] = []
    for name, s in stats.items():
        out_features = int(s.get("out_features", 0) or 0)
        in_features = int(s.get("in_features", 0) or 0)
        if out_features <= 0 or in_features <= 0:
            continue

        layer = _layer_of(name)
        role = _role_of(name)
        packed_group, expert_id = _packed_group_of(name, layer)
        if expert_id is None and s.get("expert_id") is not None:
            expert_id = int(s["expert_id"])

        vectors = {k: _as_vector(s.get(k)) for k in VECTOR_KEYS}
        vectors.update({
            k: _as_expert_array(s.get(k), 1 if k == "expert_tokens" else 2)
            for k in EXPERT_KEYS
        })
        # Shape agreement is the wiring check: these arrays index by expert on
        # dim 0 and by the SAME channel axis the unit declares on dim 1. A
        # transposed or stale array would price a real A-side against the wrong
        # channels and still look plausible, so refuse instead of pricing it.
        n_e = int(s.get("num_experts", 0) or 0)
        expected = {
            "expert_g_sq_sum": (n_e, out_features),
            "expert_act_sq_sum": (n_e, in_features),
            "expert_act_absmax": (n_e, in_features),
            "expert_tokens": (n_e,),
        }
        for k, want in expected.items():
            got = vectors.get(k)
            if got is not None and tuple(got.shape) != want:
                raise ValueError(
                    f"{name}: {k} has shape {tuple(got.shape)}, expected "
                    f"{want} from the probe's num_experts/out_features/"
                    f"in_features -- refusing to build a card on it")

        units.append(SensitivityUnit(
            topology=UnitTopology(
                name=name,
                layer_index=layer,
                role=role,
                fused_group=_fused_group_of(name, role, layer),
                packed_group=packed_group,
                expert_id=expert_id,
                source_dtype=s.get("source_dtype") or meta.get("source_dtype"),
            ),
            out_features=out_features,
            in_features=in_features,
            n_params=int(s.get("n_params", out_features * in_features)),
            n_tokens=global_tokens,
            h_trace_raw=float(s.get("h_trace_raw", 0.0) or 0.0),
            h_w2_sum_raw=float(s.get("h_w2_sum_raw", 0.0) or 0.0),
            w_norm_sq=float(s.get("w_norm_sq", 0.0) or 0.0),
            w_max_abs=float(s.get("w_max_abs", 0.0) or 0.0),
            route_prob=(float(s["route_prob"])
                        if s.get("route_prob") is not None else None),
            **vectors,
        ))

    provenance = CardProvenance(
        model_id=model_id or str(meta.get("model") or "unknown"),
        calib_hash=_calib_hash(meta),
        n_calib_samples=int(meta.get("nsamples", 0) or 0),
        seq_len=int(meta.get("seqlen", 0) or 0),
        # A probe payload made by the incremental/sharded driver may not carry
        # its producer commit.  Falling back to this card-builder checkout is
        # not evidence of what produced the Fisher: the two can be unrelated
        # revisions, and the existing GLM card demonstrated the failure by
        # persisting ``probe_commit="unknown"``.  Campaign drivers that have a
        # separately stamped probe receipt can bind that exact producer here.
        probe_commit=str(
            probe_commit or meta.get("git_commit") or _git_commit()
        ),
        render_basis=render_basis,
        notes=notes or (
            "converted from probe.pkl; no per-channel vectors present "
            "(scalar-only card)" if not any(
                u.has_vectors for u in units) else ""),
    )
    return SensitivityCard(provenance, units)


def summarize(card: SensitivityCard) -> dict[str, Any]:
    units = card.units()
    with_vec = sum(1 for u in units if u.has_vectors)
    with_gsq = sum(1 for u in units if u.g_sq_sum is not None)
    with_expert_act = sum(1 for u in units if u.has_expert_activation_stats)
    expert_floats = sum(
        sum(np.asarray(getattr(u, k)).size
            for k in EXPERT_KEYS if getattr(u, k) is not None)
        for u in units)
    # Params, not units, is the honest denominator for "is this card
    # activation-aware": on an MoE the expert tensors ARE the model.
    aqua_params = sum(
        u.n_params for u in units
        if u.g_sq_sum is not None or u.has_expert_activation_stats)
    vec_floats = sum(
        sum(np.asarray(getattr(u, k)).size
            for k in VECTOR_KEYS if getattr(u, k) is not None)
        for u in units)
    return {
        "schema_version": card.schema_version,
        "model_id": card.provenance.model_id,
        "calib_hash": card.provenance.calib_hash,
        "render_basis": card.provenance.render_basis.value,
        "fingerprint": card.provenance.fingerprint()[:16],
        "n_units": len(units),
        "n_units_with_vectors": with_vec,
        "n_units_aqua_ready": with_gsq,
        "n_units_with_expert_act_stats": with_expert_act,
        "expert_vector_floats": expert_floats,
        "expert_vector_mb_float32": round(expert_floats * 4 / 1e6, 2),
        "n_fused_groups": len(card.fused_groups()),
        "n_packed_groups": len(card.packed_groups()),
        "vector_floats": vec_floats,
        "vector_mb_float32": round(vec_floats * 4 / 1e6, 2),
        "quantizable_params": sum(u.n_params for u in units),
        "aqua_priceable_params": aqua_params,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="prismaquant.sensitivity_card_build",
        description="Build / inspect a shareable PrismaQuant Sensitivity Card.")
    ap.add_argument("probe", help="path to probe.pkl, or to an existing .npz card")
    ap.add_argument("-o", "--out", help="write the card to this .npz path")
    ap.add_argument("--model-id", default=None)
    ap.add_argument(
        "--probe-commit", default=None,
        help="exact commit from the probe producer receipt; overrides a "
             "missing probe meta.git_commit instead of misattributing the "
             "card-builder checkout to the Fisher producer")
    ap.add_argument("--render-basis", default="rtn", choices=[b.value for b in RenderBasis],
                    help="how a consumer should render weight error. 'rtn' is the "
                         "only basis a shareable card can promise: compensated "
                         "renders need per-Linear Hessians, which are not shippable. "
                         "RTN-vs-compensated dW is immaterial at fp4 but ~+36%% at fp8.")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip invariant checks (not recommended)")
    args = ap.parse_args(argv)

    if args.probe.endswith(".npz"):
        card = SensitivityCard.from_npz(args.probe)
    else:
        card = card_from_probe(
            args.probe, model_id=args.model_id,
            probe_commit=args.probe_commit,
            render_basis=RenderBasis(args.render_basis), notes=args.notes)

    if not args.no_validate:
        card.validate()

    print(json.dumps(summarize(card), indent=2))

    if args.out:
        card.to_npz(args.out)
        size = os.path.getsize(args.out)
        print(f"\nwrote {args.out}  ({size / 1e6:.2f} MB on disk)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
