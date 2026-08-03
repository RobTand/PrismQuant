#!/usr/bin/env python3
"""Summarise the two DP variants: does the DP use mid rungs, and what do they buy?

Reports, per variant: the expert layer map, the BODY split (the question the
incidental K12-K18 body columns exist to answer), Delta-loss against the
5812.11 shippable baseline, bytes, and the serving lanes every unit rides.

Also runs the MXFP8 dominance check. That one is decided by arithmetic rather
than by the DP, and deliberately so: MXFP8_E4M3 is registered W8A8
(``act_bits=8``), so even with exact weights on a block-128 source the
allocator REFUSES to short-circuit it to zero -- an activation-quantizing
format with no measured ``output_mse`` is excluded from the menu rather than
priced at the DP's global minimum. So the DP cannot be asked to confirm the
comparison; what it can do is show the mask record, which is the mechanism.
The dominance itself is unambiguous: 8.25 bpw versus 8.00049 bpw at the same
zero weight error is 3.1% more bytes for nothing.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SHIPPABLE_BASELINE = 5812.11
ALL_CB_BASELINE = 6553.987851366602
N_BODY_ALLOCATABLE = 301          # probe inventory; 89 further F8_E4M3 tensors
N_BODY_FLOOR_ONLY = 89            # are not probeable and ship source-format


def fmt_of(cfg: dict) -> str:
    if not isinstance(cfg, dict):
        return str(cfg)
    dt, k = cfg.get("data_type"), cfg.get("cb_k")
    if dt == "nvfp4_cb":
        return f"NVFP4_CB_K{k}"
    if dt == "fp8_cb":
        return f"FP8_CB_K{k}"
    if dt == "fp4_e2m1":
        return "MXFP4_SOURCE"
    if dt == "fp8_e4m3" and cfg.get("scale_fmt") == "ue8m0":
        return "FP8_BLOCK_UE8M0_SOURCE"
    return str(dt)


def summarise(root: Path, tag: str) -> dict:
    lc = json.loads((root / "layer_config.json").read_text())
    sel = json.loads((root / "selection.json").read_text())
    experts: dict[int, set[str]] = defaultdict(set)
    body: Counter[str] = Counter()
    for name, cfg in lc.items():
        if name == "__prismaquant__":
            continue
        f = fmt_of(cfg)
        m = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.", name)
        if m:
            experts[int(m.group(1))].add(f)
        else:
            body[f] += 1
    by_fmt: dict[str, list[int]] = defaultdict(list)
    for layer in sorted(experts):
        fmts = sorted(experts[layer])
        assert len(fmts) == 1, f"layer {layer} is not format-uniform: {fmts}"
        by_fmt[fmts[0]].append(layer)
    dloss = float(sel["predicted_dloss"])
    mid_rungs = {f for f in by_fmt if re.fullmatch(r"NVFP4_CB_K(1[2367]|18)", f)}
    return {
        "variant": tag,
        "expert_layer_map": {k: v for k, v in sorted(by_fmt.items())},
        "expert_uses_mid_rungs": sorted(mid_rungs),
        "n_expert_layers_on_mid_rungs": sum(
            len(by_fmt[f]) for f in mid_rungs),
        "body_split": dict(body.most_common()),
        "body_allocatable": N_BODY_ALLOCATABLE,
        "body_floor_only_not_allocatable": N_BODY_FLOOR_ONLY,
        "predicted_dloss": dloss,
        "vs_shippable_baseline_pct": round(
            100 * (1 - dloss / SHIPPABLE_BASELINE), 3),
        "vs_all_cb_baseline_pct": round(
            100 * (1 - dloss / ALL_CB_BASELINE), 3),
        "bytes": {
            k: sel.get(k) for k in (
                "predicted_whole_artifact_upper_bound_gb",
                "predicted_floor_gb",
                "predicted_body_tensor_payload_gb",
                "selection_headroom_gb",
                "chosen_achieved_bits",
            )
        },
        "lane_units": sel.get("serving_lane_provenance", {}).get(
            "activation_contracts", {}),
    }


def mxfp8_dominance() -> dict:
    """8.25 bpw MXFP8 re-encode vs 8.00049 bpw source passthrough."""
    import prismaquant.format_registry as fr

    shape = (8192, 4096)                      # layers.N.attn.wo_a
    mx = fr.get_format("MXFP8_E4M3")
    src = fr.get_format("FP8_BLOCK_UE8M0_SOURCE")
    mx_b = mx.memory_bytes_for_shape(shape)
    src_b = src.memory_bytes_for_shape(shape)
    return {
        "question": "can an MXFP8 re-encode ever beat the source passthrough?",
        "answer": "no -- strictly dominated",
        "mxfp8_bpw": round(mx.effective_bits, 5),
        "passthrough_bpw": round(src.effective_bits, 5),
        "bytes_on_wo_a": {"mxfp8": mx_b, "passthrough": src_b},
        "extra_bytes_pct": round(100 * (mx_b / src_b - 1), 3),
        "both_zero_weight_error_on_block128_source": True,
        "why_the_dp_cannot_confirm_it": (
            "MXFP8_E4M3 is registered W8A8 (act_bits=8), so "
            "cost_entry_is_bit_exact refuses to short-circuit it even with "
            "exact weights, and with no measured output_mse "
            "cost_entry_prices_unmeasured_activation_at_zero EXCLUDES it from "
            "the menu rather than pricing an unmeasured activation cost at "
            "the DP's global minimum. The exclusion is the correct behaviour, "
            "not a gap: the comparison is settled by bytes at equal error."
        ),
    }


def main() -> None:
    probe = Path(sys.argv[1] if len(sys.argv) > 1 else
                 "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/"
                 "artifacts-mxfp4/probe-k12k18")
    out = {
        "schema": "prismaquant.dsv4_mxfp4_dual_allocation.v1",
        "baselines": {
            "all_cb": ALL_CB_BASELINE,
            "shippable_mxfp4_experts_only": SHIPPABLE_BASELINE,
        },
        "mxfp8_dominance": mxfp8_dominance(),
    }
    for tag, note in (
        ("a", "conservative fallback: no body passthrough, for a deploy that "
              "cannot set GRIDBOOK_MXFP8_DENSE=1"),
        ("b", "LIKELY SHIP PICK: gridbook 0.8.0 dense MXFP8 lane is released "
              "and correctness-audited, so route_status is 'backed'. OPT-IN "
              "pending its native-parity TIMING bench, so it makes "
              "GRIDBOOK_MXFP8_DENSE=1 load-bearing alongside marlin."),
    ):
        root = probe / f"alloc-{tag}"
        if (root / "selection.json").is_file():
            out[f"variant_{tag}"] = {**summarise(root, tag), "note": note}
        else:
            out[f"variant_{tag}"] = {"status": "not run", "note": note}
    merge_report = probe / "merge_report.json"
    if merge_report.is_file():
        out["interpolator_validation"] = json.loads(
            merge_report.read_text()).get("validation")
    (probe / "DUAL_RESULT.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
