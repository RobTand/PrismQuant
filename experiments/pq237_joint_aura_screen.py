#!/usr/bin/env python3
"""Bounded fixed-teacher GPU screen; never a serving or promotion receipt.

Run in the project's known GPU container. ProductionWeightCache owns renders;
PerturbedActivationCache owns capture and candidate forwards. The only extra
hooks are the independent direct-output-residual oracle for the production
joint projection lease. No assignment is used to recenter the teacher.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prismaquant import format_registry as fr
from prismaquant.kl_fisher import fisher_probe_scalar
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache, _activation_qdq, activation_cache_filename,
)
from prismaquant.production_weight_cache import ProductionWeightCache, render_production_weight
from prismaquant.tessera_hessian import calibration_identity


CALIBRATION = [
    "A river carries water from high ground toward the sea. Rainfall and melting snow feed its tributaries. Engineers measure the water level to understand seasonal flooding and plan bridges. The valley changes slowly as sediment accumulates on the river banks. People living nearby use the water for irrigation and transport. The observations were recorded every morning at the same station.",
    "A computer program can sort a collection by comparing pairs of values. A stable sorting method preserves the original order of equal elements. Testing includes empty inputs, repeated values, and lists that are already ordered. The programmer records both the expected output and the reason each example matters. Clear interfaces allow one implementation to replace another without changing the surrounding application.",
]
HELDOUT = [
    "The astronomer adjusted the telescope after sunset. Several exposures revealed a faint companion star near the brighter source. To distinguish a real signal from sensor noise, the team repeated the observation on another night. Their report describes the instrument, exposure time, and uncertainty of the estimated position. Independent observations can establish whether the object follows the predicted orbit.",
    "Bread dough rises when yeast consumes sugars and releases carbon dioxide. The baker controls temperature and resting time to develop a suitable texture. Flour absorbs water differently depending on its protein content. A recipe gives useful starting proportions, but the dough must also be observed during mixing. After baking, the loaf cools on a rack before it is sliced for the evening meal.",
    "A city council considered plans for a new public garden. Residents proposed shaded paths, native flowers, and space for children to play. The designers estimated construction expenses and future maintenance needs. They compared several layouts using the same plot of land. After reviewing comments, the council chose a proposal and published a schedule for the work to begin in spring.",
    "The museum catalog describes a ceramic bowl discovered near an ancient settlement. Its painted surface shows a repeating geometric pattern. Researchers compare the clay with samples from nearby hills to estimate where the object was made. The date remains uncertain because the excavation record is incomplete. The display explains both the evidence and the limits of the current interpretation.",
]
FORMATS = ("TESSERA_E2M1_K1_R768", "TESSERA_E4M3_K1_R896", "TESSERA_BF16_K1_R896")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensor_digest(tensor):
    return digest(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def paired_summary(left, right):
    """Probe uncertainty of a *paired difference*, not independent error bars."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired summary requires at least two aligned probes")
    values = [0.5 * (float(a)**2 - float(b)**2) for a, b in zip(left, right)]
    mean = sum(values) / len(values)
    stderr = math.sqrt(sum((x - mean)**2 for x in values) / (len(values) - 1) / len(values))
    return {"mean_difference": mean, "probe_stderr": stderr, "per_probe_difference": values}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 7, 21])
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--sequence-length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=237000)
    ap.add_argument("--tessera-commit", required=True)
    ap.add_argument("--source-manifest", required=True,
                    help="Host git revision/status and tracked-file SHA256 map, verified in container")
    args = ap.parse_args()
    if args.probes < 2 or args.sequence_length < 2 or not 1 <= len(args.layers) <= 4:
        raise ValueError("screen bounds require >=2 probes/tokens and 1..4 units")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    start = time.time()
    source = json.loads(Path(args.source_manifest).read_text())
    root = Path(__file__).resolve().parents[1]
    for relative, expected in source["files"].items():
        if file_digest(root / relative) != expected:
            raise ValueError(f"source changed after manifest: {relative}")
    for relative, expected in source.get("symlinks", {}).items():
        if os.readlink(root / relative) != expected:
            raise ValueError(f"source symlink changed after manifest: {relative}")
    identity = {
        "schema": "prismaquant.pq237.numerical_screen.v1", "args": vars(args),
        "source": source,
        "harness_sha256": file_digest(__file__), "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "model_files": {p.name: file_digest(p) for p in sorted(Path(args.model).iterdir()) if p.is_file()},
        "start_unix": start, "calibration_text": CALIBRATION, "heldout_text": HELDOUT,
        "scope": "resident numerical QDQ screen; no served latency, no promotion",
        "execution_authority": "User explicitly prohibited PrismaBuild for this session; direct known Docker image",
    }
    dump(out / "identity.json", identity)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    def tokenize(texts):
        # Texts are sufficiently long; refuse padding rather than price pad tokens.
        rows = [tokenizer(x, add_special_tokens=False)["input_ids"] for x in texts]
        if any(len(row) < args.sequence_length for row in rows):
            raise ValueError("text is shorter than declared unpadded sequence length")
        return torch.tensor([row[:args.sequence_length] for row in rows], device="cuda")
    calib, heldout = tokenize(CALIBRATION), tokenize(HELDOUT)
    dump(out / "tokens.json", {"calibration": calib.tolist(), "heldout": heldout.tolist(),
                               "calibration_sha256": tensor_digest(calib), "heldout_sha256": tensor_digest(heldout)})
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="eager").cuda().eval()
    model.config.use_cache = False
    model.requires_grad_(False)
    names = [f"model.layers.{layer}.mlp.down_proj" for layer in args.layers]
    modules = {name: model.get_submodule(name) for name in names}
    # One early trainable target keeps every subsequent cotangent reachable.
    for module in modules.values():
        module.weight.requires_grad_(True)
    capture = PerturbedActivationCache(model, dict.fromkeys(names, "BF16"), out / "activations",
        input_rows=calib.numel(), cal_hash=tensor_digest(calib), include_activation_quant=False)
    capture.install()
    with torch.no_grad():
        model(calib)
    capture.remove()
    capture_receipt = capture.finalize()
    if capture_receipt["missing"] or set(capture_receipt["written"]) != set(names):
        raise RuntimeError(f"activation coverage differs: {capture_receipt}")
    activations = {name: torch.load(out / "activations" / activation_cache_filename(name),
        weights_only=True)["inputs"].cuda() for name in names}
    levers = {"tessera_hessian_identity": calibration_identity("\n".join(CALIBRATION),
        [calib], fit_tokens=calib.numel())}
    cache = ProductionWeightCache(weights={}, levers=levers,
        activation_max_abs={name: float(x.abs().max()) for name, x in activations.items()})
    # Observe the existing byte renderer so the receipt binds the real blob.
    import prismaquant.tessera_render as tr
    encode = tr.encode_tessera_unit
    emitted = {}
    def record_encode(weight, fmt, **kwargs):
        render, blob = encode(weight, fmt, **kwargs)
        emitted.update(blob_bytes=len(blob), blob_sha256=digest(blob))
        return render, blob
    tr.encode_tessera_unit = record_encode
    render_rows = {}
    try:
        for name, module in modules.items():
            render_rows[name] = {}
            for fmt in FORMATS:
                emitted.clear()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with torch.no_grad():
                    rendered = render_production_weight(module.weight, fmt, qname=name,
                        activations=activations, levers=levers)
                torch.cuda.synchronize()
                cache.weights[(name, fmt)] = rendered
                render_rows[name][fmt] = {**emitted, "render_seconds": time.perf_counter() - begin,
                    "render_sha256": tensor_digest(rendered), "source_sha256": tensor_digest(module.weight),
                    "shape": list(rendered.shape), "activation_bits": fr.get_format(fmt).act_bits,
                    "registry_bits": float(fr.get_format(fmt).bits_for_shape(rendered.shape))}
                print("RENDER", name, fmt, render_rows[name][fmt], flush=True)
                dump(out / "renders.json", render_rows)
    finally:
        tr.encode_tessera_unit = encode
    del activations
    from prismaquant.joint_aura import SignedJointProjectionLease, activation_identity, arithmetic_identity
    delta = {(name, fmt): cache.get(name, fmt).float() - module.weight.float()
             for name, module in modules.items() for fmt in FORMATS}
    specs = {name: {fmt: fr.get_format(fmt) for fmt in FORMATS} for name in names}
    identity["joint_source_sha256"] = file_digest(Path(__file__).resolve().parents[1] / "prismaquant/joint_aura.py")
    identity["arithmetic"] = arithmetic_identity(torch.bfloat16)
    identity["activation_contracts"] = {name: {fmt: activation_identity(spec, cache.activation_max_abs, name)
        for fmt, spec in specs[name].items()} for name in names}
    identity["resident_render_bytes"] = sum(w.numel() * w.element_size() for w in cache.weights.values())
    dump(out / "identity.json", identity)
    probes, oracle = [], {}
    def oracle_hook(name):
        def forward(module, inputs, output):
            x = inputs[0].detach()
            def backward(g):
                with torch.no_grad():
                    for fmt in FORMATS:
                        spec = specs[name][fmt]
                        xq = _activation_qdq(x, spec, cache.activation_max_abs, name) if spec.act_quant_changes_input else x
                        residual = xq.float() @ cache.get(name, fmt).float().T - x.float() @ module.weight.float().T
                        oracle[(name, fmt)] = float((g.float() * residual).sum())
                return g
            output.register_hook(backward)
        return forward
    lease = SignedJointProjectionLease(modules, specs, delta, activation_max_abs=cache.activation_max_abs)
    handles = [module.register_forward_hook(oracle_hook(name)) for name, module in modules.items()]
    with lease:
        for index in range(args.probes):
            model.zero_grad(set_to_none=True)
            lease.begin_probe()
            logits = model(calib).logits
            fisher_probe_scalar(logits, seed=args.seed + index, token_scope="causal",
                                distribution="rademacher").backward()
            terms = lease.finish_probe()
            probes.append(terms)
            if index == 0:
                for handle in handles:
                    handle.remove()
                errors = [{"unit": name, "format": fmt, "direct": oracle[(name, fmt)],
                    "decomposed": terms[(name, fmt)]["total"],
                    "absolute_error": abs(oracle[(name, fmt)] - terms[(name, fmt)]["total"])}
                    for name in names for fmt in FORMATS]
                dump(out / "oracle.json", errors)
                if any(x["absolute_error"] > 2e-5 + 2e-4 * abs(x["direct"]) for x in errors):
                    raise AssertionError("joint decomposition disagrees with direct residual oracle")
            print("PROBE", index + 1, "of", args.probes, flush=True)
            del logits
    dump(out / "probes.json", [{name: {fmt: probe[(name, fmt)] for fmt in FORMATS}
                                for name in names} for probe in probes])
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    del delta
    with torch.no_grad():
        teacher = model(heldout).logits[:, :-1].float()
        teacher_logp = teacher.log_softmax(-1)
        teacher_p = teacher_logp.exp()
        teacher_nll = -teacher_logp.gather(-1, heldout[:, 1:, None]).squeeze(-1).mean(-1)
    del teacher
    measurements = []
    assignments = list(itertools.product(FORMATS, repeat=len(names)))
    def evaluate(choice):
        hook = PerturbedActivationCache(model, dict(zip(names, choice)), out / "unused",
            input_rows=0, cal_hash=tensor_digest(calib), production_weight_cache=cache, capture_inputs=False)
        hook.install()
        try:
            with torch.no_grad(), hook.frozen_weight_cache(), hook.materialized_frozen_weights():
                z = model(heldout).logits[:, :-1].float().log_softmax(-1)
                kl = (teacher_p * (teacher_logp - z)).sum(-1).mean(-1)
                nll = -z.gather(-1, heldout[:, 1:, None]).squeeze(-1).mean(-1)
        finally:
            hook.remove()
        return {"kl_per_text": kl.tolist(), "mean_kl": float(kl.mean()),
                "nll_per_text": nll.tolist(), "mean_nll": float(nll.mean())}
    for choice in assignments:
        row = {"assignment": dict(zip(names, choice)),
            "bytes": sum(render_rows[name][fmt]["blob_bytes"] for name, fmt in zip(names, choice)),
            "joint_additive": sum(0.5 * sum(p[(name, fmt)]["total"]**2 for p in probes) / len(probes)
                                  for name, fmt in zip(names, choice)),
            "weight_additive": sum(0.5 * sum(p[(name, fmt)]["weight"]**2 for p in probes) / len(probes)
                                   for name, fmt in zip(names, choice)),
            "signed_joint": [sum(p[(name, fmt)]["total"] for name, fmt in zip(names, choice)) for p in probes]}
        row.update(evaluate(choice))
        measurements.append(row)
        print("ASSIGNMENT", len(measurements), "/", len(assignments), "KL", row["mean_kl"], flush=True)
        dump(out / "assignments.json", measurements)
    minimum, maximum = min(r["bytes"] for r in measurements), max(r["bytes"] for r in measurements)
    budget = minimum + (maximum - minimum) // 2  # fixed policy; uses bytes only
    feasible = [r for r in measurements if r["bytes"] <= budget]
    baseline = min(feasible, key=lambda r: (r["weight_additive"], r["bytes"]))
    candidate = min(feasible, key=lambda r: (r["joint_additive"], r["bytes"]))
    oracle_best = min(feasible, key=lambda r: (r["mean_kl"], r["bytes"]))
    neighbors = []
    for left, right in itertools.combinations(measurements, 2):
        changed = [name for name in names if left["assignment"][name] != right["assignment"][name]]
        if len(changed) != 1:
            continue
        actual = left["mean_kl"] - right["mean_kl"]
        weight = left["weight_additive"] - right["weight_additive"]
        joint = left["joint_additive"] - right["joint_additive"]
        neighbors.append({"left": left["assignment"], "right": right["assignment"],
            "changed_unit": changed[0], "heldout_kl_difference": actual,
            "weight_additive_difference": weight, "joint_additive_difference": joint,
            "weight_order_correct": actual * weight > 0,
            "joint_order_correct": actual * joint > 0,
            "pricing_crossover": weight * joint < 0})
    dump(out / "neighbors.json", neighbors)
    profile_windows = {}
    for label, selected in [("weight_only", baseline), ("joint", candidate)]:
        choice = [selected["assignment"][name] for name in names]
        # Profile identical steady-state numerical workloads after one warm-up.
        evaluate(choice)
        profile_start = time.time()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(3):
                evaluate(choice)
        prof.export_chrome_trace(str(out / f"{label}-profile.json"))
        (out / f"{label}-profile.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
        profile_windows[label] = {"start_unix": profile_start, "end_unix": time.time(), "forwards": 3}
    for name, module in modules.items():
        if tensor_digest(module.weight) != render_rows[name][FORMATS[0]]["source_sha256"]:
            raise AssertionError(f"assignment evaluation did not restore teacher weight: {name}")
    summary = {"budget_bytes": budget, "budget_rule": "min + floor((max-min)/2), bytes only",
        "quantizable_parameters": sum(m.weight.numel() for m in modules.values()),
        "baseline": baseline, "candidate": candidate, "heldout_oracle_diagnostic": oracle_best,
        "baseline_regret": baseline["mean_kl"] - oracle_best["mean_kl"],
        "candidate_regret": candidate["mean_kl"] - oracle_best["mean_kl"],
        "paired_candidate_minus_baseline": paired_summary(candidate["signed_joint"], baseline["signed_joint"]),
        "teacher_nll_per_text": teacher_nll.tolist(), "projection_telemetry": lease.telemetry,
        "assignment_count": len(measurements), "feasible_count": len(feasible),
        "neighbor_pairs": len(neighbors), "pricing_crossovers": sum(r["pricing_crossover"] for r in neighbors),
        "weight_neighbor_order_correct": sum(r["weight_order_correct"] for r in neighbors),
        "joint_neighbor_order_correct": sum(r["joint_order_correct"] for r in neighbors),
        "teacher_selected_weights_restored": True, "profile_windows": profile_windows,
        "end_unix": time.time(), "wall_seconds": time.time() - start,
        "limitations": ["Three selected units only; other weights remain BF16.",
            "Authored calibration and heldout texts are a bounded numerical screen, not a benchmark dataset.",
            "Probe stderr does not represent calibration generalization uncertainty.",
            "Heldout oracle is retrospective and is never used to choose the proposed assignment.",
            "QDQ model execution is not an actual served operator route; no TTFT/decode/runtime price supplied.",
            "No sparse-anchor fit, second model, serving smoke, downstream task suite or promotion gate."]}
    dump(out / "summary.json", summary)
    print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
