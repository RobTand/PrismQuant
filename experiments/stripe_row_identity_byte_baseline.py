"""Byte proof for PrismaQuant #130: does a narrowed render keep the full run's rows?

The production cache hooks Linears off ONE ``torch.Generator``
(``production_weight_cache._LinearActivationCollector``), so the slice of the
priority stream a given Linear receives is a function of how many rows every
*earlier* hook consumed. ``56c765d`` made the draw happen for every **hooked**
Linear, which fixes ``--resume`` (that narrows the *store* set, not the hook
set). Anything that narrows the **hook** set was out of its reach.

Three narrowings reached the hook set:

  1. ``build_production_cache.py`` ``--include-qnames-file``, i.e. every
     stripe, which shortened the ``qnames`` list before the fill call.
  2. ``production_weight_cache.py`` ``qname_set = set(render_formats_by_qname)``
     fed to the collector as the hook set -- i.e. every ``--render-scope
     assignment`` build, which is the shipping default (``run-pipeline.sh:590``).
  3. ``store_qnames`` -- resume. Fixed by ``56c765d``.

(1) and (2) are fixed on this branch: the collector now hooks the caller's whole
``eligible_qnames`` enumeration and narrowing arrives as ``render_assignment``
or the new ``render_qnames``. The invariant that buys, and that sections [4]-[6]
hold to: **the bytes of a (qname, fmt) pair are a function of ``qnames`` and the
calibration, never of which subset this call renders.** Section [2] keeps
pricing the one narrowing that still changes bytes -- shortening ``qnames``
itself -- because that is now a documented caller error rather than the way a
stripe is expressed.

Reading a diff is not evidence. This renders. Everything is held identical
across arms except the narrowing: one frozen state_dict, one frozen
calibration, GPTQ on, NVFP4 out, ``cache_dir=None`` so the arms cannot resume
off each other.

Sections:
  [0] the reservoir must be selecting, or an agreement proves nothing
  [1] the render must be deterministic, or a divergence proves nothing
  [2] the pre-fix convention: a shortened ``qnames`` still moves the bytes
  [3] the N_CALIB sweep -- is the cause the bin SHAPE or the shared stream?
  [4] ``--render-scope assignment``, the shipping default -- now identical
  [5] the stripe as a ``render_qnames`` narrowing -- now identical
  [6] the real shape: ONE BF16 unit, last in the enumeration
  [7] how much do real production assignments narrow the hook set?

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
    python experiments/stripe_row_identity_byte_baseline.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import prismaquant.production_weight_cache as pwc  # noqa: E402

HIDDEN = 64
LAYERS = 4
N_CALIB = 3
SEQLEN = 40
MAX_ACT_ROWS = 16          # < N_CALIB * SEQLEN, so the reservoir must SELECT
FORMAT = "NVFP4"
LEVERS = {"gptq": True, "static_act_order": True, "joint_scale_opt": False}


class _Attn(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.o_proj(torch.tanh(self.q_proj(x)))


class _Model(nn.Module):
    """Minimal HF-shaped model: takes input_ids, returns ``.logits``."""

    def __init__(self, hidden: int = HIDDEN, layers: int = LAYERS) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, hidden)
        self.layers = nn.ModuleList([_Attn(hidden) for _ in range(layers)])

    def forward(self, input_ids, use_cache=False):
        del use_cache
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(logits=x)


def all_qnames(layers: int = LAYERS) -> list[str]:
    return [
        f"layers.{i}.{leaf}"
        for i in range(layers)
        for leaf in ("q_proj", "o_proj")
    ]


def layer_qnames(indices, layers: int = LAYERS) -> list[str]:
    want = {int(i) for i in indices}
    return [q for q in all_qnames(layers) if int(q.split(".")[1]) in want]


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def frozen_state() -> dict:
    torch.manual_seed(0)
    return {k: v.clone() for k, v in _Model().state_dict().items()}


def frozen_calib(n_calib: int = N_CALIB) -> torch.Tensor:
    torch.manual_seed(0)
    _ = _Model()  # consume the same stream prefix frozen_state() does
    return torch.randint(0, 64, (n_calib, SEQLEN), dtype=torch.long)


REAL_ASSIGNMENTS = (
    ("qwen38-27b-scout20", "/home/rob/dq-runs/qwen38-27b-scout20/artifacts/layer_config.json"),
    ("prod-27b-nvfp4cb-5p5", "/home/rob/dq-runs/prod-27b-nvfp4cb-5p5/artifacts/layer_config.json"),
    ("fc45-0p6b-nvfp4", "/home/rob/dq-runs/fc45-0p6b-nvfp4/artifacts/layer_config.json"),
)


def real_assignment_census() -> list[dict]:
    """How much does a REAL layer_config.json narrow the hook set?

    ``--render-scope assignment`` drops every BF16 entry from the set the
    collector hooks (``production_weight_cache.py`` :4888/:5182 pre-fix), so
    the BF16 count IS the narrowing. A toy shows the mechanism; this says
    whether production ever exercises it. Records the file digest so the row
    is checkable.
    """
    from prismaquant.production_recache import _load_assignment

    rows: list[dict] = []
    for label, path in REAL_ASSIGNMENTS:
        f = Path(path)
        if not f.is_file():
            continue
        row: dict = {
            "label": label,
            "path": str(f),
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        }
        try:
            assignment = _load_assignment(str(f))
            units = len(assignment)
            bf16 = sum(
                1 for fmt in assignment.values()
                if str(fmt).strip().upper() == "BF16"
            )
            row.update({
                "units": units,
                "bf16": bf16,
                "non_bf16": units - bf16,
                "narrowed_pct": (100.0 * bf16 / units) if units else 0.0,
            })
        except Exception as exc:  # a stale/foreign config must not abort
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def run_arm(
    state: dict,
    calib: torch.Tensor,
    qnames,
    *,
    render_assignment=None,
    render_qnames=None,
    max_act_rows: int = MAX_ACT_ROWS,
) -> dict:
    """Render a subset of ``qnames``; digest the rows AND the rendered bytes.

    ``qnames`` is the enumeration the collector hooks. ``render_assignment``
    and ``render_qnames`` narrow only what gets rendered. Shortening
    ``qnames`` is the pre-fix calling convention and is what section [2]
    prices.
    """
    model = _Model()
    model.load_state_dict(state)
    model.eval()

    captured: dict[str, torch.Tensor] = {}
    orig_render = pwc.render_production_weight

    def spy(weight, fmt, *, qname, activations, **kwargs):
        rows = activations.get(qname)
        if rows is not None and qname not in captured:
            captured[qname] = rows.detach().cpu().clone()
        return orig_render(
            weight, fmt, qname=qname, activations=activations, **kwargs
        )

    pwc.render_production_weight = spy
    try:
        cache = pwc.fill_production_weight_cache(
            model, calib, list(qnames),
            formats=[FORMAT],
            render_assignment=render_assignment,
            render_qnames=render_qnames,
            levers=LEVERS,
            max_act_rows=max_act_rows,
            cache_dir=None,          # no resume between arms
            progress=False,
        )
    finally:
        pwc.render_production_weight = orig_render

    metadata = cache.metadata or {}
    return {
        "weights": {
            q: digest(t) for (q, f), t in cache.weights.items() if f == FORMAT
        },
        "rows": {q: digest(t) for q, t in captured.items()},
        "row_shapes": {q: tuple(t.shape) for q, t in captured.items()},
        "hook_scope": metadata.get("activation_hook_scope"),
    }


def compare(got: dict, ref: dict) -> tuple[list[str], list[str]]:
    rows = [q for q in sorted(got["rows"]) if got["rows"][q] != ref["rows"].get(q)]
    byts = [
        q for q in sorted(got["weights"])
        if got["weights"][q] != ref["weights"].get(q)
    ]
    return rows, byts


def _line(label: str, got: dict, ref: dict) -> dict:
    rows, byts = compare(got, ref)
    n = len(got["weights"])
    verdict = "IDENTICAL to unstriped" if not byts else "DIVERGES"
    print(f"  {label:<50} rows {len(rows)}/{n}  bytes {len(byts)}/{n}   {verdict}")
    return {"units": n, "rows_differ": rows, "bytes_differ": byts}


def main() -> int:
    state = frozen_state()
    calib = frozen_calib()
    full_names = all_qnames()
    stripe_np = layer_qnames([1, 3])   # non-prefix -- the LPT bin shape
    stripe_p = layer_qnames([0, 1])    # prefix -- a contiguous partition

    print("=" * 80)
    print("PrismaQuant #130 -- does a narrowed render keep the full run's rows?")
    print(f"  model      : {LAYERS} layers x 2 Linears, hidden={HIDDEN}")
    print(f"  calibration: {N_CALIB} x {SEQLEN} = {N_CALIB * SEQLEN} rows/Linear")
    print(f"  max_act_rows={MAX_ACT_ROWS}  format={FORMAT}")
    print(f"  levers={LEVERS}")
    print("=" * 80)

    control = run_arm(state, calib, full_names)
    reference = run_arm(state, calib, full_names)

    # [0] With no selection pressure every arm agrees for a reason that has
    #     nothing to do with the bug, and the whole comparison is vacuous.
    kept = sorted({int(s[0]) for s in reference["row_shapes"].values()})
    selecting = kept == [MAX_ACT_ROWS]
    print(f"\n[0] reservoir selects: kept {kept} of {N_CALIB * SEQLEN} "
          f"candidate rows -> {selecting}")
    if not selecting:
        print("    ABORT: no selection pressure; the comparison proves nothing.")
        return 2

    # [1] Without determinism a divergence is unreadable.
    deterministic = control["weights"] == reference["weights"]
    print(f"[1] render is deterministic (control == reference): {deterministic}")
    if not deterministic:
        print("    ABORT: render nondeterminism; no arm comparison is readable.")
        return 2

    print("\n[2] the PRE-FIX convention -- the qnames ARGUMENT itself "
          "narrowed, which shortens the hook set")
    summary = {
        "stripe_nonprefix_L1L3": _line(
            "non-prefix stripe (layers 1,3) -- the LPT shape",
            run_arm(state, calib, stripe_np), reference),
        "stripe_prefix_L0L1": _line(
            "prefix stripe (layers 0,1) -- a contiguous partition",
            run_arm(state, calib, stripe_p), reference),
    }

    # [3] The discriminator. Within ONE forward a prefix subset's hooks fire at
    #     the same stream offsets as the full run's. A second calibration sample
    #     re-enters the stream behind a shorter hook set, and then even the
    #     model's FIRST Linear gets a different slice. If the prefix arm
    #     diverges at N_CALIB >= 2, the cause is the shared stream, not the bin
    #     shape -- and a contiguous partition is no smaller a perturbation.
    print("\n[3] is the cause the bin SHAPE or the shared stream? sweep N_CALIB")
    sweep = []
    for n in (1, 2, 3):
        c = frozen_calib(n)
        ref_n = run_arm(state, c, full_names)
        _, p_bytes = compare(run_arm(state, c, stripe_p), ref_n)
        _, n_bytes = compare(run_arm(state, c, stripe_np), ref_n)
        row = {
            "n_calib": n,
            "prefix_bytes_differ": len(p_bytes),
            "nonprefix_bytes_differ": len(n_bytes),
            "units": len(stripe_p),
        }
        sweep.append(row)
        print(f"    N_CALIB={n}: prefix stripe {len(p_bytes)}/{len(stripe_p)} "
              f"units differ, non-prefix {len(n_bytes)}/{len(stripe_np)}")

    print("\n[4] --render-scope assignment -- the SHIPPING default "
          "(run-pipeline.sh:590). Same qnames argument in both arms; the "
          "narrowing arrives only as render_assignment, so the hook set is "
          "held at the full enumeration and the bytes must match.")
    assigned = {
        q: (FORMAT if q in set(stripe_np) else "BF16") for q in full_names
    }
    summary["render_scope_assignment"] = _line(
        "assignment scope vs format-menu scope",
        run_arm(state, calib, full_names, render_assignment=assigned),
        reference)

    print("\n[5] the stripe, expressed as a render narrowing rather than an "
          "enumeration narrowing (build_production_cache.py render_qnames)")
    summary["stripe_via_render_qnames_nonprefix"] = _line(
        "non-prefix stripe (layers 1,3) via render_qnames",
        run_arm(state, calib, full_names, render_qnames=stripe_np),
        reference)
    summary["stripe_via_render_qnames_prefix"] = _line(
        "prefix stripe (layers 0,1) via render_qnames",
        run_arm(state, calib, full_names, render_qnames=stripe_p),
        reference)

    # [6] The real-assignment shape. A production assignment does not carve
    #     the model in half -- it BF16s a handful of units, and the ones it
    #     BF16s can be anywhere. The worst case for "surely a few units cannot
    #     matter" is a single BF16 unit LAST in the enumeration: within one
    #     forward pass every earlier hook fires at the same stream offset, so
    #     a single-sample run would agree. It is the second calibration sample
    #     that re-enters the stream behind a shorter hook set.
    print("\n[6] the real shape -- ONE BF16 unit, LAST in the enumeration")
    one_bf16 = {q: FORMAT for q in full_names[:-1]}
    one_bf16[full_names[-1]] = "BF16"
    summary["one_bf16_last_old_convention"] = _line(
        "pre-fix convention: qnames shortened to the 7 rendered",
        run_arm(state, calib, full_names[:-1]), reference)
    summary["one_bf16_last_fixed"] = _line(
        "fixed: full enumeration hooked, assignment narrows the render",
        run_arm(state, calib, full_names, render_assignment=one_bf16),
        reference)

    # [7] How much do REAL assignments narrow the hook set? A toy proves the
    #     mechanism; this prices it. Skipped (not failed) when the work dirs
    #     are absent, so the harness stays runnable off this box.
    print("\n[7] real assignments -- how much would the pre-fix hook set "
          "have been narrowed?")
    census = real_assignment_census()
    for row in census:
        if row.get("error"):
            print(f"  {row['label']:<28} unreadable: {row['error']}")
            continue
        print(f"  {row['label']:<28} {row['non_bf16']:>4}/{row['units']:<4} "
              f"hooked  narrowed by {row['bf16']} "
              f"({row['narrowed_pct']:.1f}%)")
    if not census:
        print("  (no layer_config.json found on this box -- section skipped)")

    out = {
        "schema": "prismaquant.stripe_row_identity_byte_baseline.v2",
        "settings": {
            "layers": LAYERS, "hidden": HIDDEN, "n_calib": N_CALIB,
            "seqlen": SEQLEN, "max_act_rows": MAX_ACT_ROWS,
            "format": FORMAT, "levers": LEVERS,
        },
        "gates": {
            "reservoir_selects": selecting,
            "render_deterministic": deterministic,
        },
        "reference_digests": reference,
        "n_calib_sweep": sweep,
        "real_assignment_census": census,
        "summary": summary,
    }
    dest = REPO_ROOT / "experiments" / "results" / "pq130_stripe_row_identity.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
