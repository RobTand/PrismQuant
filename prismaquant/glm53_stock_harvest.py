"""GLM-5.3-Flash dense-half production-anchor AURA harvest (the GPU seam).

This is the batched streamed KL-adjoint harvest that
``glm53_stock_reprice.ANCHOR_MEASUREMENT_CONTRACT`` names: AURA ``gW`` is a
live-autograd quantity, so the harvest stays batched (one capture pass + one
reverse layer-major adjoint pass over the whole dense plan) and this module's
output is consumed as *finished scalars* by ``glm53_stock_reprice campaign``
via ``anchors_from_measured_scalars``.  The two halves are deliberately
separate processes: this one owns the GPU and the 306 GiB streamed source;
the campaign action is CPU-only pricing/merge and can run anywhere the
artifacts are readable.

Scope
-----
Dense quantizable units only (``unit_class == "dense"`` from
``build_declarations``): the costed rung is rendered in the fixed production
arm (GPTQ + static_act_order + JSO -- ``stock_anchored_cost.RENDER_LEVERS``)
and measured under the streamed KL adjoint.  Routed packed experts are
excluded on purpose (``include_routed_experts=False`` +
``allow_packed_expert_omission=True``): the smooth AURA cost is
route-flip-blind for them, and their serving-unit KL is measured by
``expert_empirical_cost`` in its own currency.  Passthrough terminals
(FP8_SOURCE / BF16) are never rendered here -- they are priced exactly at
zero by the campaign's pinned/terminal rows.

Identity
--------
On a CB-free plan the renderer identity binds no levers by itself, so the
production arm identity built here is value-bearing: render levers, costed
format, serving profile, calibration provenance (both the probe calibration
hash that produced the activation rows and the observed tokenized hash --
asserted equal before any GPU work), and the AURA probe parameters.  The
wrapper payload written by this module carries that arm identity, the exact
plan, and the streamed model identity next to the raw AURA payload, because
the AURA payload alone does not restate the renderer identity.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time

HARVEST_SCHEMA = "prismaquant.glm53_stock_harvest.v1"
ARM_SCHEMA = "prismaquant.glm53_stock_harvest.arm.v1"

# Everything below glm53_stock_reprice is CPU-safe to import: its CPU-only
# device mask is applied in its main(), not at module import.
from prismaquant.glm53_stock_reprice import (
    CHECKPOINT_CENSUS_SCHEMA,
    MODEL_PROFILE,
    PROBE_CENSUS_SCHEMA,
    SERVING_PROFILE,
    Glm53StockError,
    build_declarations,
    load_census,
)
from prismaquant.stock_anchored_cost import (
    DEFAULT_COSTED_FORMAT,
    RENDER_LEVERS,
)


def _log(msg: str) -> None:
    print(f"[glm53-harvest {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def refuse_volatile_path(label: str, path: str | Path, repo: Path) -> Path:
    """/tmp was cleared by an OOM and took a set of artifacts with it."""
    resolved = Path(path).resolve()
    if str(resolved).startswith("/tmp") or str(resolved).startswith(
        str(repo.resolve())
    ):
        raise Glm53StockError(
            f"{label} {resolved} resolves under /tmp or the repo; refusing "
            "a durable artifact location that an OOM clear or a tree sync "
            "can destroy"
        )
    return resolved


def build_dense_plan(
    probe_census: Mapping[str, object],
    checkpoint_census: Mapping[str, object],
    *,
    unit_filter: str | None = None,
    max_units: int = 0,
) -> dict[str, object]:
    """The exact per-qname render plan for the dense half.

    Returns a dict with ``plan`` ({qname: (costed_format,)}), the packed
    routed-expert unit names deliberately excluded, the pinned units, the
    ladder refusals, and ``plan_scope`` -- ``"full"`` unless a filter or cap
    narrowed it, in which case the output is stamped partial so a smoke run
    can never masquerade as the campaign harvest.
    """
    declarations, pinned, refusals, unresolved = build_declarations(
        probe_census, checkpoint_census,
    )
    if unresolved:
        raise Glm53StockError(
            f"{len(unresolved)} probe unit(s) have no checkpoint census "
            f"entry, e.g. {unresolved[:5]}; a unit that belongs to no "
            "provenance is a unit nothing prices"
        )
    dense = [d for d in declarations if d.unit_class == "dense"]
    unknown_classes = sorted({
        d.unit_class for d in declarations
        if d.unit_class not in ("dense", "packed_expert")
    })
    if unknown_classes:
        raise Glm53StockError(
            f"unknown unit classes {unknown_classes}; this harvest knows "
            "exactly two and refuses to guess a third"
        )
    # Packed routed experts are classified from the probe census, NOT from
    # the surviving declarations: on this checkpoint every packed unit loses
    # its FP8_SOURCE terminal to the serving profile (profile_mismatch) and
    # lands in refusals before reaching a declaration.  They are priced by
    # the empirical serving-unit KL path; the refusals stay in the output as
    # the recorded serving-gap finding.
    probe_units = probe_census.get("units")
    if not isinstance(probe_units, Mapping):
        raise Glm53StockError("probe census has no units")
    packed = sorted(
        str(qname) for qname, row in probe_units.items()
        if isinstance(row, Mapping) and row.get("is_packed")
    )
    declared = {d.qname for d in declarations}
    refused = {item.qname for item in refusals}
    lost = [
        name for name in packed
        if name not in declared and name not in refused
    ]
    if lost:
        raise Glm53StockError(
            f"{len(lost)} packed unit(s) belong to neither declarations nor "
            f"refusals, e.g. {lost[:3]}; the partition accounting is broken"
        )
    plan_scope = "full"
    if unit_filter:
        pattern = re.compile(unit_filter)
        dense = [d for d in dense if pattern.search(d.qname)]
        plan_scope = "filtered"
    if max_units and max_units > 0 and len(dense) > max_units:
        dense = dense[:max_units]
        plan_scope = "filtered"
    if not dense:
        raise Glm53StockError(
            "dense plan is empty; a harvest with nothing to render is a "
            "mis-specified filter, not a fast success"
        )
    plan = {d.qname: (d.costed_format,) for d in dense}
    return {
        "plan": plan,
        "plan_scope": plan_scope,
        "unit_filter": unit_filter or None,
        "max_units": int(max_units or 0),
        "dense_units": sorted(plan),
        "packed_expert_units_excluded": packed,
        "pinned_units": {name: list(value) for name, value in pinned.items()},
        "ladder_refusals": [item.to_dict() for item in refusals],
    }


def build_arm_identity(
    *,
    probe_calib_hash: str,
    observed_calib_hash: str,
    dataset: str,
    n_calib_samples: int,
    calib_seqlen: int,
    calib_seed: int,
    n_probes: int,
    max_act_rows: int,
) -> dict[str, object]:
    """The value-bearing production arm identity for a CB-free plan.

    The streamed checkpoint identity binds the anchor renderer's exact
    identity, but on a CB-free plan the renderer itself binds no levers --
    they enter only through this arm identity.  A run with changed levers,
    a different costed format, or a different calibration must therefore
    produce a different identity and refuse a stale resume.
    """
    if probe_calib_hash != observed_calib_hash:
        raise Glm53StockError(
            f"tokenized calibration hash {observed_calib_hash} differs from "
            f"the probe's {probe_calib_hash}; the activation rows were "
            "captured under the probe calibration and a mismatched render "
            "calibration would be a rendering confound"
        )
    return {
        "schema": ARM_SCHEMA,
        "campaign": "glm53-flash-stock-anchored",
        "model_profile": MODEL_PROFILE,
        "serving_profile_id": SERVING_PROFILE,
        "costed_format": DEFAULT_COSTED_FORMAT,
        "render_levers": dict(RENDER_LEVERS),
        "calibration": {
            "dataset": str(dataset),
            "n_samples": int(n_calib_samples),
            "seqlen": int(calib_seqlen),
            "seed": int(calib_seed),
            "calib_hash": str(observed_calib_hash),
            "probe_calib_hash": str(probe_calib_hash),
        },
        "aura": {
            "n_probes": int(n_probes),
            "token_scope": "all",
            "temperature": 1.0,
            "seed_base": 7000,
            "dw_dtype": "bfloat16",
            "max_act_rows": int(max_act_rows),
        },
    }


def _precheck_activation_files(
    act_dir: Path, qnames: Sequence[str],
) -> None:
    """Seconds-cheap existence check before the ~8 min model load.

    The renderer's own fail-closed activation coverage gate remains the
    authority; this just moves the common failure (a missing or wrong act
    dir) from minute 9 to second 1.
    """
    sub = re.compile(r"[^A-Za-z0-9_-]")
    missing = [
        name for name in qnames
        if not (act_dir / (sub.sub("__", name) + ".pt")).is_file()
    ]
    if missing:
        raise Glm53StockError(
            f"{len(missing)} planned unit(s) have no activation file under "
            f"{act_dir}, e.g. {missing[:5]}; the probe's --activation-cache-"
            "dir must cover every rendered unit"
        )


def run_harvest(args: argparse.Namespace) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[1]
    checkpoint_dir = refuse_volatile_path(
        "--checkpoint-dir", args.checkpoint_dir, repo,
    )
    output = refuse_volatile_path("--output", args.output, repo)
    if not args.dataset:
        raise Glm53StockError(
            "--dataset is required; the loader's default is a live Hub "
            "stream, not the campaign corpus"
        )

    from prismaquant.gpu_guard import require_cuda_hot_path
    device = require_cuda_hot_path("glm53_stock_harvest", args.device)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    probe_census = load_census(args.probe_census, schema=PROBE_CENSUS_SCHEMA)
    checkpoint_census = load_census(
        args.checkpoint_census, schema=CHECKPOINT_CENSUS_SCHEMA,
    )
    probe_meta = probe_census.get("meta")
    if not isinstance(probe_meta, Mapping) or not probe_meta.get("calib_hash"):
        raise Glm53StockError("probe census meta carries no calib_hash")
    probe_calib_hash = str(probe_meta["calib_hash"])

    planned = build_dense_plan(
        probe_census, checkpoint_census,
        unit_filter=args.unit_filter, max_units=args.max_units,
    )
    plan: dict[str, tuple[str, ...]] = planned["plan"]
    _log(
        f"plan: {len(plan)} dense unit(s) x {DEFAULT_COSTED_FORMAT} "
        f"(scope={planned['plan_scope']}, "
        f"{len(planned['packed_expert_units_excluded'])} packed expert(s) "
        f"excluded for the empirical path, "
        f"{len(planned['pinned_units'])} pinned)"
    )

    act_dir = Path(args.activation_cache_dir)
    _precheck_activation_files(act_dir, sorted(plan))

    import torch
    from transformers import AutoTokenizer
    from prismaquant.build_rtn_cache import stage_multimodal
    from prismaquant.cost_streaming import (
        build_streamed_causal_lm,
        build_streamed_model_identity,
    )
    from prismaquant.measure_quant_cost import ActivationIndex
    from prismaquant.model_profiles import detect_profile
    from prismaquant.perturbed_x_cache import calibration_data_hash
    from prismaquant.sensitivity_probe import load_calibration

    staged, _cleanup = stage_multimodal(args.model)
    profile = detect_profile(staged)
    if profile.name != MODEL_PROFILE:
        raise Glm53StockError(
            f"detected profile {profile.name!r} is not {MODEL_PROFILE!r}; "
            "this harvest is checkpoint-specific on purpose"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        staged, trust_remote_code=True,
        local_files_only=Path(staged).exists(),
    )
    calibration = load_calibration(
        tokenizer, args.dataset, args.n_calib_samples, args.calib_seqlen,
        calib_seed=args.calib_seed,
    )
    observed_calib_hash = calibration_data_hash(calibration)
    arm_identity = build_arm_identity(
        probe_calib_hash=probe_calib_hash,
        observed_calib_hash=observed_calib_hash,
        dataset=args.dataset,
        n_calib_samples=args.n_calib_samples,
        calib_seqlen=args.calib_seqlen,
        calib_seed=args.calib_seed,
        n_probes=args.n_probes,
        max_act_rows=args.max_act_rows,
    )
    calibration = calibration.to(device)
    _log(f"calibration ready {tuple(calibration.shape)} hash={observed_calib_hash}")

    activation_index = ActivationIndex(act_dir, sorted(plan))

    _log(f"loading streamed {args.model} (staged={staged}) bf16 ...")
    runner = build_streamed_causal_lm(
        staged,
        device=device,
        dtype=torch.bfloat16,
        offload_folder=str(checkpoint_dir / "streamed-model-offload"),
        profile=profile,
    )
    try:
        model_identity = build_streamed_model_identity(
            runner,
            args.model,
            identity_cache_path=checkpoint_dir / "streamed_model_identity.json",
        )
        from prismaquant.aura_cost import run_streamed_production_anchor_aura

        payload = run_streamed_production_anchor_aura(
            runner,
            calibration,
            formats_by_qname=plan,
            activation_index=activation_index,
            render_levers=dict(RENDER_LEVERS),
            col_weights={},
            cb_serialization_context=None,
            calibration_hash=probe_calib_hash,
            arm_identity=arm_identity,
            model_identity=model_identity,
            checkpoint_dir=checkpoint_dir / "aura",
            resume=bool(args.resume),
            n_probes=int(args.n_probes),
            max_act_rows=int(args.max_act_rows),
            include_routed_experts=False,
            allow_packed_expert_omission=True,
            profile=profile,
            checkpoint_identity_extra={
                "campaign_schema": HARVEST_SCHEMA,
                "plan_scope": planned["plan_scope"],
                "serving_profile_id": SERVING_PROFILE,
            },
        )
    finally:
        runner.shutdown()

    wrapper = {
        "schema": HARVEST_SCHEMA,
        "plan_scope": planned["plan_scope"],
        "unit_filter": planned["unit_filter"],
        "max_units": planned["max_units"],
        "plan": {name: list(fmts) for name, fmts in sorted(plan.items())},
        "packed_expert_units_excluded": planned["packed_expert_units_excluded"],
        "pinned_units": planned["pinned_units"],
        "ladder_refusals": planned["ladder_refusals"],
        "arm_identity": arm_identity,
        "model_identity": dict(model_identity),
        "checkpoint_dir": str(checkpoint_dir),
        "aura_payload": payload,
    }
    from prismaquant.cost_stage_checkpoint import atomic_write_bytes

    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        output, pickle.dumps(wrapper, protocol=pickle.HIGHEST_PROTOCOL),
    )
    provenance = payload.get("provenance", {})
    side = output.with_suffix(".provenance.json")
    side.write_text(json.dumps({
        "schema": HARVEST_SCHEMA,
        "output": str(output),
        "plan_scope": planned["plan_scope"],
        "dense_units": len(plan),
        "anchor_rows": int(provenance.get("dw_production_anchor_rows", 0)),
        "rtn_fallback_rows": int(provenance.get("dw_rtn_fallback_rows", 0)),
        "git_commit": provenance.get("git_commit"),
        "calib_hash": observed_calib_hash,
        "n_probes": int(args.n_probes),
    }, indent=2, sort_keys=True) + "\n")
    _log(
        f"wrote {output}: {len(payload.get('costs', {}))} unit(s), "
        f"{provenance.get('dw_production_anchor_rows')} production-anchor "
        f"row(s), sidecar {side.name}"
    )
    return wrapper


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prismaquant.glm53_stock_harvest",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe-census", required=True)
    parser.add_argument("--checkpoint-census", required=True)
    parser.add_argument("--activation-cache-dir", required=True)
    parser.add_argument(
        "--dataset", required=True,
        help="Pinned calibration corpus. Required: the probe's activation "
             "rows bind the calibration, and the loader default is a live "
             "Hub stream.",
    )
    parser.add_argument("--n-calib-samples", type=int, required=True)
    parser.add_argument("--calib-seqlen", type=int, required=True)
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--n-probes", type=int, default=32)
    parser.add_argument(
        "--max-act-rows", type=int, default=256,
        help="Per-unit activation row cap for the production render; must "
             "not exceed what the probe cached.",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--unit-filter", default=None,
        help="Regex narrowing the dense plan (smoke runs). The output is "
             "stamped plan_scope=filtered and the campaign refuses it.",
    )
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_harvest(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
