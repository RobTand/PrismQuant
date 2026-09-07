#!/usr/bin/env python3
"""Predeclared dense fixed-teacher qualification of the streamed joint currency.

This consumes an immutable protocol prepared without model inference. It uses
the production caches, compute_aura_cost_streamed, durable unit checkpoints,
and allocator candidate construction. It does not produce serving prices.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import time


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_source_manifest(root, path):
    """Refuse any source under `root` the manifest does not exactly account for.

    "Exact file closure" is two claims, and only the first used to be checked:
    every listed entry still has its recorded kind and content, and no unlisted
    entry under the root can be reached by the import system. The second is the
    shared policy in experiments/pq237_source_closure.py, which the writer
    (experiments/pq237_source_manifest.py) enumerates with the same function.
    """
    from experiments.pq237_source_closure import verify_source_closure

    root = Path(root)
    manifest = json.loads(Path(path).read_text())
    if not manifest.get("files"):
        raise ValueError("source manifest requires an exact file closure")
    # `excluded_symlinks` records links whose target lies outside the root, so
    # their content cannot be sealed. Declaring one keeps it out of `files`; it
    # never exempts it from the closure, and its kind and target are checked
    # exactly as a listed symlink's are.
    declared_symlinks = dict(manifest.get("symlinks", {}))
    declared_symlinks.update(manifest.get("excluded_symlinks", {}))
    for relative, expected in manifest["files"].items():
        entry = root / relative
        # Kind before content: file_sha256 reads through a symlink, so a
        # regular file swapped for a link would otherwise verify silently.
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"source entry is no longer a regular file: {relative}")
        if file_sha256(entry) != expected:
            raise ValueError(f"source changed after manifest: {relative}")
    for relative, expected in declared_symlinks.items():
        entry = root / relative
        if not entry.is_symlink():
            raise ValueError(f"source entry is no longer a symlink: {relative}")
        if os.readlink(entry) != expected:
            raise ValueError(f"source symlink changed after manifest: {relative}")
    verify_source_closure(root, manifest["files"], declared_symlinks)
    return manifest


def load_candidate_payload(path, plan, operator_bindings, probe_sha256):
    """Validate persisted producer currency against independently held bindings.

    Pickles here are our own local producer artifacts, never untrusted input.
    The exact row roster must survive the production candidate constructor.
    """
    from prismaquant import format_registry as fr
    from prismaquant.allocator_candidates import build_candidates
    from prismaquant.cost_currency import require_run_currency
    from prismaquant.joint_aura import identity_sha256

    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    costs = payload["costs"]
    coordinates = {(name, fmt) for name, formats in plan.items() for fmt in formats}
    actual = {(name, fmt) for name, rows in costs.items() for fmt in rows}
    if actual != coordinates or set(payload["stats"]) != set(plan):
        raise ValueError("persisted joint coordinate scope differs from frozen plan")
    provenance = payload["provenance"]
    if (identity_sha256(provenance["probe_identity"]) != probe_sha256
            or provenance["probe_identity_sha256"] != probe_sha256):
        raise ValueError("persisted joint probe identity differs from measured input")
    if (identity_sha256(provenance["joint_aura_identity"])
            != provenance["joint_aura_identity_sha256"]):
        raise ValueError("persisted joint provenance digest mismatch")
    if (identity_sha256(provenance["joint_aura_identity"]["probe_identity"])
            != probe_sha256):
        raise ValueError("persisted joint provenance probe mismatch")
    currency = require_run_currency(payload)
    for name, fmt in sorted(coordinates):
        row = costs[name][fmt]
        if row["joint_operator_identity_sha256"] != operator_bindings[name][fmt]:
            raise ValueError(f"persisted joint operator binding differs: {name}@{fmt}")
        if row["probe_identity_sha256"] != probe_sha256:
            raise ValueError("persisted joint row probe identity mismatch")
        shape = row["joint_operator_identity"]["source_weight"]["shape"]
        stats = payload["stats"][name]
        if (len(shape) != 2 or stats.get("n_params") != math.prod(shape)
                or stats.get("in_features") != shape[1] or stats.get("out_features") != shape[0]):
            raise ValueError("persisted joint stats differ from the bound dense source shape")
    masks, menu = [], {}
    candidates = build_candidates(
        payload["stats"], costs,
        [fr.get_format(fmt) for fmt in sorted({fmt for _, fmt in coordinates})],
        mask_records=masks, tessera_menu_report=menu,
        preserve_runtime_frontier=True,
    )
    admitted = {(name, candidate.fmt) for name, rows in candidates.items() for candidate in rows}
    if admitted != coordinates:
        raise ValueError(f"production candidate coordinate scope differs: masks={masks}, "
                         f"missing={sorted(coordinates - admitted)}")
    for name, rows in candidates.items():
        for candidate in rows:
            if candidate.predicted_dloss != costs[name][candidate.fmt]["predicted_dloss"]:
                raise ValueError("candidate changed mean joint currency; require UCB z=0 and no gains")
    return payload, candidates, {
        "payload_sha256": file_sha256(path), "currency": currency,
        "mask_records": masks, "menu": menu,
        "candidate_mode": "research mean currency; no runtime table or solver selection",
        "candidates": {name: [asdict(candidate) for candidate in rows]
                       for name, rows in candidates.items()},
    }


def _probe_summary(values):
    mean = sum(values) / len(values)
    return {"mean_difference": mean,
            "paired_standard_error": math.sqrt(sum((v - mean)**2 for v in values)
                                                / (len(values) - 1) / len(values)),
            "difference_per_probe": values,
            "uncertainty_scope": "probe_sampling_conditional_on_fixed_calibration"}


def unit_activation_source_recorder(captured_source, activation_source):
    """Record only the activation source built for ``captured_source["unit"]``.

    The encoder also builds unit-less sources while encoding (an empty one and
    an identity probe, ``tessera_render._encoder_kwargs_for_plane``), AFTER
    the renderer built the unit's own source. Keeping the last source handed
    the probe to the wire receipt, which then refused for lack of an exact
    Hessian key (PQ #261). Only the source carrying THIS unit's Hessian is the
    production activation source the receipt must name.
    """
    def record(hessians, *args, **kwargs):
        source = activation_source(hessians, *args, **kwargs)
        if captured_source.get("unit") in hessians:
            captured_source["value"] = source
        return source
    return record


def retain_production_wire(weight, rendered, blob, *, qname, fmt, activation_source, wire_dir):
    """Retain the original bytes using the existing campaign/producer grammar."""
    from prismaquant.production_weight_cache import _cb_cache_tensor_identity
    from prismaquant.tessera_campaign import _wire_path
    from prismaquant.tessera_formats import parse_tessera_format_name
    from tessera import cached_unit
    from tessera.unit_artifact import read_unit_artifact

    family, rung = parse_tessera_format_name(fmt)
    input_identity = cached_unit.encoding_input_identity(
        weight, qname, family.payload_grid(), int(rung), activation=activation_source)
    path = _wire_path(Path(wire_dir), qname, fmt)
    record = cached_unit.make_unit_record(blob, input_identity, filename=path.name)
    temporary = path.with_suffix(".tessera.tmp")
    temporary.write_bytes(blob)
    os.replace(temporary, path)
    persisted = path.read_bytes()
    cached_unit.verify_cached_unit(persisted, record, input_identity)
    decoded = read_unit_artifact(persisted, device=str(rendered.device)).to(rendered.dtype)
    decoded_identity = _cb_cache_tensor_identity(decoded)
    if decoded_identity != _cb_cache_tensor_identity(rendered):
        raise ValueError("retained wire does not decode to the production cache tensor")
    return {"blob_bytes": len(blob), "blob_sha256": record["blob_sha256"],
            "wire_record": record, "wire_path": str(path.relative_to(Path(wire_dir).parent)),
            "wire_decoded_weight": decoded_identity}


def compare_assignments(payload, candidates, assignments):
    from prismaquant.joint_aura import assignment_probe_summary, paired_assignment_difference

    costs = payload["costs"]
    choices = {key: {name: costs[name][fmt] for name, fmt in assignment.items()}
               for key, assignment in assignments.items()}
    scores = {}
    for key, rows in choices.items():
        # Validate the complete row roster before using its weight components.
        additive = assignment_probe_summary(rows, objective="additive")
        count = len(next(iter(rows.values()))["probe_ids"])
        weight = [sum(0.5 * row["signed_components_per_probe"][k]["weight"]**2
                      for row in rows.values()) for k in range(count)]
        scores[key] = {
            "assignment": assignments[key], "joint_additive": additive,
            "joint_quadratic_diagnostic": assignment_probe_summary(rows, objective="joint_quadratic"),
            "weight_component_additive_diagnostic": {
                "mean": sum(weight) / count, "per_probe": weight,
                "role": "same-render weight component; not a joint allocator row"},
            "allocator_mean_cost": sum(next(c.predicted_dloss for c in candidates[name]
                                             if c.fmt == fmt)
                                       for name, fmt in assignments[key].items()),
        }
    pairs = {}
    for background in ("A8", "A16"):
        left, right = f"L0{background}_L21A4", f"L0{background}_L21A16"
        additive = paired_assignment_difference(choices[left], choices[right], objective="additive")
        weight_left = scores[left]["weight_component_additive_diagnostic"]["per_probe"]
        weight_right = scores[right]["weight_component_additive_diagnostic"]["per_probe"]
        pairs[background] = {
            "direction": f"{left} minus {right}", "joint_additive": additive,
            "joint_quadratic_diagnostic": paired_assignment_difference(
                choices[left], choices[right], objective="joint_quadratic"),
            "weight_component_additive_diagnostic": {
                **_probe_summary([a - b for a, b in zip(weight_left, weight_right)]),
                "probe_identity_sha256": additive["probe_identity_sha256"],
                "role": "diagnostic weight component; not a joint allocator row"},
        }
    return scores, pairs


def _capture_and_render(model, calibration, plan, out, *, calibration_text):
    import torch
    from prismaquant import format_registry as fr
    from prismaquant.perturbed_x_cache import PerturbedActivationCache, activation_cache_filename
    from prismaquant.production_weight_cache import ProductionWeightCache, render_production_weight, _cb_cache_tensor_identity
    from prismaquant.tessera_hessian import calibration_identity
    import prismaquant.tessera_render as render_owner
    import prismaquant.tessera_hessian as hessian_owner

    modules = {name: model.get_submodule(name) for name in plan}
    capture = PerturbedActivationCache(
        model, dict.fromkeys(plan, "BF16"), out / "activations",
        input_rows=calibration.numel(), cal_hash=_cb_cache_tensor_identity(calibration)["content_sha256"],
        include_activation_quant=False,
    )
    capture.install()
    try:
        with torch.no_grad():
            reference_logits = model(calibration).logits.detach().cpu()
    finally:
        capture.remove()
    capture_receipt = capture.finalize()
    if capture_receipt["missing"] or set(capture_receipt["written"]) != set(plan):
        raise RuntimeError(f"activation capture coverage differs: {capture_receipt}")
    activations = {name: torch.load(out / "activations" / activation_cache_filename(name),
                                   weights_only=True)["inputs"].cuda() for name in plan}
    levers = {"tessera_hessian_identity": calibration_identity(
        calibration_text, [calibration], fit_tokens=calibration.numel())}
    cache = ProductionWeightCache(weights={}, levers=levers,
                                  activation_max_abs={name: float(x.abs().max())
                                                      for name, x in activations.items()})
    rows, emitted, captured_source = {}, {}, {}
    wire_dir = out / "wire"
    wire_dir.mkdir()
    encode = render_owner.encode_tessera_unit
    activation_source = hessian_owner.activation_source

    record_activation_source = unit_activation_source_recorder(captured_source, activation_source)

    def record_encode(weight, fmt, **kwargs):
        rendered, blob = encode(weight, fmt, **kwargs)
        source = captured_source.get("value")
        if bool(kwargs.get("activation_kwargs")) != (source is not None):
            raise ValueError("wire receipt lost the actual production activation source")
        emitted.update(retain_production_wire(weight, rendered, blob,
            qname=name, fmt=fmt, activation_source=source, wire_dir=wire_dir))
        return rendered, blob

    render_owner.encode_tessera_unit = record_encode
    hessian_owner.activation_source = record_activation_source
    try:
        for name, formats in plan.items():
            rows[name] = {}
            source = modules[name].weight.detach()
            for fmt in formats:
                emitted.clear()
                captured_source.clear()
                captured_source["unit"] = name
                with torch.no_grad():
                    rendered = render_production_weight(source, fmt, qname=name,
                                                        activations=activations, levers=levers)
                if set(emitted) != {"blob_bytes", "blob_sha256", "wire_record", "wire_path", "wire_decoded_weight"}:
                    raise RuntimeError("production renderer did not expose exactly one serialized blob")
                cache.weights[name, fmt] = rendered
                rows[name][fmt] = {**emitted, "rendered_weight": _cb_cache_tensor_identity(rendered),
                                  "source_weight": _cb_cache_tensor_identity(source),
                                  "activation_bits": fr.get_format(fmt).act_bits}
                dump(out / "renders.json", rows)
                print("RENDER", name, fmt, emitted["blob_bytes"], flush=True)
    finally:
        render_owner.encode_tessera_unit = encode
        hessian_owner.activation_source = activation_source
        captured_source.clear()
    # The small five-render cache is deliberately resident and is persisted as
    # the actual ProductionWeightCache object. There is no shadow render store.
    cache_path = out / "production.pkl"
    with cache_path.open("wb") as stream:
        pickle.dump(cache, stream, protocol=pickle.HIGHEST_PROTOCOL)
    del cache, activations
    with cache_path.open("rb") as stream:
        cache = pickle.load(stream)
    return cache, rows, capture_receipt, reference_logits


def _direct_oracle(model, calibration, cache, plan, payload):
    """Independent one-probe dense dY, compared to the real streamed rows."""
    import torch
    from prismaquant import format_registry as fr
    from prismaquant.kl_fisher import fisher_probe_scalar
    from prismaquant.perturbed_x_cache import _activation_qdq

    modules = {name: model.get_submodule(name) for name in plan}
    captures, handles = {}, []
    for name, module in modules.items():
        module.weight.requires_grad_(True)

        def record(mod, inputs, output, name=name):
            output.retain_grad()
            captures.setdefault(name, []).append((inputs[0].detach(), output))

        handles.append(module.register_forward_hook(record))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(calibration).logits
        identity = payload["provenance"]["probe_identity"]
        fisher_probe_scalar(logits, seed=identity["seed_base"], token_scope="causal",
                            distribution="rademacher", temperature=1.0).backward()
        errors = []
        with torch.no_grad():
            for name, formats in plan.items():
                source = modules[name].weight.float()
                for fmt in formats:
                    spec = fr.get_format(fmt)
                    rendered = cache.get(name, fmt).float()
                    direct = 0.0
                    for x, output in captures[name]:
                        qx = (_activation_qdq(x, spec, cache.activation_max_abs, name)
                              if spec.act_quant_changes_input else x)
                        residual = qx.float() @ rendered.T - x.float() @ source.T
                        direct += float((output.grad.float() * residual).sum())
                    measured = payload["costs"][name][fmt]["signed_per_probe"][0]
                    errors.append({"unit": name, "format": fmt, "direct": direct,
                                   "streamed": measured, "absolute_error": abs(direct - measured),
                                   "tolerance": 2e-5 + 2e-4 * abs(direct)})
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
    return errors


@contextmanager
def _assignment(model, assignment, cache, out):
    from prismaquant.perturbed_x_cache import PerturbedActivationCache

    owner = PerturbedActivationCache(model, assignment, out / "no-capture",
                                    input_rows=0, cal_hash="qualification",
                                    production_weight_cache=cache, capture_inputs=False)
    owner.install()
    try:
        with owner.frozen_weight_cache(), owner.materialized_frozen_weights():
            yield
    finally:
        owner.remove()


def evaluate_splits(model, cache, protocol, out):
    import torch
    from prismaquant.production_weight_cache import _cb_cache_tensor_identity
    from experiments.pq237_streamed_protocol import paired_sequence_summary

    assignments = protocol["assignments"]
    source = {name: _cb_cache_tensor_identity(model.get_submodule(name).weight)
              for name in protocol["plan"]}
    measurements, contrasts = {}, {}
    with torch.no_grad():
        for split, sequences in protocol["splits"].items():
            rows = {name: {} for name in ("teacher", *assignments)}
            # Two sequences per forward bounds full-vocabulary FP32 temporaries.
            for start in range(0, len(sequences), 2):
                batch = sequences[start:start + 2]
                ids = torch.tensor([row["tokens"] for row in batch], device="cuda", dtype=torch.long)
                teacher_logp = model(ids).logits[:, :-1].float().log_softmax(-1)
                teacher_p = teacher_logp.exp()
                teacher_nll = -teacher_logp.gather(-1, ids[:, 1:, None]).squeeze(-1).mean(-1)
                for index, sequence in enumerate(batch):
                    rows["teacher"][sequence["sequence_id"]] = {"kl": 0.0, "nll": float(teacher_nll[index])}
                for name, assignment in assignments.items():
                    profile_context = (torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True, profile_memory=True)
                        if split == "original_calibration" and start == 0 else nullcontext(None))
                    with profile_context as profile:
                        with _assignment(model, assignment, cache, out):
                            logp = model(ids).logits[:, :-1].float().log_softmax(-1)
                    if profile is not None:
                        profile.export_chrome_trace(str(out / f"{name}.trace.json"))
                        (out / f"{name}.profile.txt").write_text(profile.key_averages().table(
                            sort_by="self_cuda_time_total", row_limit=40))
                    kl = (teacher_p * (teacher_logp - logp)).sum(-1).mean(-1)
                    nll = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1).mean(-1)
                    for index, sequence in enumerate(batch):
                        rows[name][sequence["sequence_id"]] = {"kl": float(kl[index]), "nll": float(nll[index])}
            measurements[split] = rows
            contrasts[split] = {}
            for background in ("A8", "A16"):
                left, right = f"L0{background}_L21A4", f"L0{background}_L21A16"
                contrasts[split][background] = {
                    metric: paired_sequence_summary(
                        {key: row[metric] for key, row in rows[left].items()},
                        {key: row[metric] for key, row in rows[right].items()})
                    for metric in ("kl", "nll")}
            # Difference of the two paired swaps retains sequence covariance.
            contrasts[split]["background_interaction"] = {
                metric: paired_sequence_summary(
                    {key: rows["L0A8_L21A4"][key][metric] - rows["L0A8_L21A16"][key][metric]
                     for key in rows["teacher"]},
                    {key: rows["L0A16_L21A4"][key][metric] - rows["L0A16_L21A16"][key][metric]
                     for key in rows["teacher"]}) for metric in ("kl", "nll")}
            dump(out / "evaluations.json", measurements)
            dump(out / "sequence-contrasts.json", contrasts)
            print("EVALUATED", split, len(sequences), flush=True)
    if source != {name: _cb_cache_tensor_identity(model.get_submodule(name).weight) for name in source}:
        raise RuntimeError("candidate evaluation failed exact source-weight restoration")
    return measurements, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--corpus-arrow", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--tessera-root", required=True)
    parser.add_argument("--tessera-source-manifest", required=True)
    parser.add_argument("--image-id", required=True, help="Exact Docker image ID used by the admitted launch")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    args = parser.parse_args()
    from experiments.pq237_streamed_protocol import load_protocol

    protocol = load_protocol(args.protocol, args.protocol_sha256, model_path=args.model,
                             corpus_arrow=args.corpus_arrow)
    root = Path(__file__).resolve().parents[1]
    historical = json.loads((root / "experiments/results/pq237-joint-aura-20260905/tokens.json").read_text())
    for split, key in (("original_calibration", "calibration"), ("original_holdout", "heldout")):
        if [row["tokens"] for row in protocol["splits"][split]] != historical[key]:
            raise ValueError(f"{split} token IDs differ from the historical screen")
    source = verify_source_manifest(root, args.source_manifest)
    tessera_source = verify_source_manifest(args.tessera_root, args.tessera_source_manifest)
    # The known image need not contain Git. The existing producer override is
    # admitted only after the source map has verified every packaged file.
    commit = source.get("commit", "")
    if len(commit) not in (40, 64) or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("verified source manifest needs an exact Git commit")
    if os.environ.get("PRISMAQUANT_IDENTITY_GIT_COMMIT", commit) != commit:
        raise ValueError("checkpoint Git override differs from verified source manifest")
    os.environ["PRISMAQUANT_IDENTITY_GIT_COMMIT"] = commit
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "protocol.json", protocol)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    import tessera
    from prismaquant.aura_cost import compute_aura_cost_streamed, _aura_source_sha256
    from prismaquant.cost_streaming import build_streamed_causal_lm, build_streamed_model_identity
    from prismaquant.joint_aura import activation_identity, arithmetic_identity, identity_sha256, prefetch_joint_cache
    from prismaquant import format_registry as fr
    from prismaquant.model_profiles import detect_profile
    from prismaquant.production_weight_cache import _cb_cache_tensor_identity

    if not torch.cuda.is_available():
        raise RuntimeError("this qualification requires an admitted GPU action")
    if not Path(tessera.__file__).resolve().is_relative_to(Path(args.tessera_root).resolve()):
        raise ValueError("imported Tessera differs from verified producer root")
    os.environ["PRISMAQUANT_COST_UCB_Z"] = "0"
    os.environ["PRISMAQUANT_TESSERA_MENU"] = "research"
    # Retain the historical rank-reversal screen's QDQ policy. Native FP8
    # serving has a separately qualified, unclipped policy; this pilot cannot
    # supply a timing price for that different activation operator.
    os.environ["PRISMAQUANT_PROD_ACT_SCALES"] = "1"
    os.environ["PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES"] = "0"
    os.environ["PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE"] = "0"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(protocol["seed_base"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    start = time.time()
    plan = protocol["plan"]
    formats = sorted({fmt for rows in plan.values() for fmt in rows})
    calibration = torch.tensor([row["tokens"] for row in protocol["splits"]["original_calibration"]],
                               device="cuda", dtype=torch.long)
    identity = {"schema": "prismaquant.pq237.streamed_qualification.v1",
                "protocol_sha256": args.protocol_sha256, "source": source,
                "tessera_source": tessera_source, "torch": torch.__version__,
                "transformers": transformers.__version__, "container_image_id": args.image_id,
                "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
                "arithmetic": arithmetic_identity(torch.bfloat16), "start_unix": start,
                "render_identity_policy": "new renders; old hashes are historical comparators only",
                "scope": "dense fixed-teacher QDQ diagnosis; no served runtime or promotion",
                "profiling_scope": "first two calibration sequences per assignment, including materialization; no warmup or latency claim",
                "execution_authority": "PrismaBuild admitted batch GPU action"}
    dump(out / "identity.json", identity)

    def resident_model():
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                    local_files_only=True, attn_implementation="eager").cuda().eval()
        model.config.use_cache = False
        model.requires_grad_(False)
        return model

    model = resident_model()
    cache, renders, capture, reference_logits = _capture_and_render(model, calibration, plan, out,
        calibration_text="\n".join(row["text"] for row in protocol["splits"]["original_calibration"]))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    identity["activation_capture"] = capture
    identity["resident_cache"] = prefetch_joint_cache(cache, plan, plan,
        max_resident_bytes=int((torch.cuda.mem_get_info()[0] / 1024**3 - args.min_free_gib) * 1024**3))
    runner = build_streamed_causal_lm(args.model, device=torch.device("cuda"), dtype=torch.bfloat16,
        offload_folder=str(out / "streamed-offload"), profile=detect_profile(args.model),
        prefetch_workers=2, prefetch_lookahead=2, require_prefetched_residency=True)
    runner.model.config._attn_implementation = "eager"
    runner.model.config.use_cache = False
    try:
        model_identity = build_streamed_model_identity(runner, args.model,
            identity_cache_path=out / "streamed-model-identity.json")
        # Bind the input contract before the producer returns any rows. Its
        # own self-consistent digest cannot authorize different probe inputs.
        expected_probe = {
            "schema": "prismaquant.joint_aura.probes.v1", "source_model": model_identity,
            "calibration_sha256": _cb_cache_tensor_identity(calibration)["content_sha256"],
            "calibration_shape": list(calibration.shape), "calibration_dtype": str(calibration.dtype),
            "n_probes": protocol["n_probes"], "seed_base": protocol["seed_base"],
            "token_scope": "causal", "temperature": 1.0, "distribution": "rademacher",
            "normalization": "global_kl_fisher", "producer_source_sha256": _aura_source_sha256(),
            "arithmetic": arithmetic_identity(torch.bfloat16),
        }
        expected_probe_sha256 = identity_sha256(expected_probe)
        expected = {name: {fmt: identity_sha256({
            "schema": "prismaquant.joint_aura.operator.v1", "qname": name, "format": fmt,
            "source_weight": renders[name][fmt]["source_weight"],
            "rendered_weight": renders[name][fmt]["rendered_weight"],
            "activation": activation_identity(fr.get_format(fmt), cache.activation_max_abs, name),
            "arithmetic": arithmetic_identity(torch.bfloat16),
            "probe_identity_sha256": expected_probe_sha256,
        }) for fmt in formats} for name, formats in plan.items()}
        dump(out / "expected-currency-bindings.json", {
            "probe": expected_probe, "operator_identity_sha256_by_candidate": expected})
        with torch.no_grad():
            actual_logits = runner(calibration).logits.detach().cpu()
        parity = {"bit_exact": torch.equal(actual_logits, reference_logits),
                  "max_absolute_logit_difference": float((actual_logits.float() - reference_logits.float()).abs().max())}
        dump(out / "teacher-parity.json", parity)
        if not parity["bit_exact"]:
            raise RuntimeError("streamed/resident BF16 teacher differs; frozen qualification refuses")
        del reference_logits, actual_logits
        kwargs = dict(n_probes=protocol["n_probes"], token_scope="causal", temperature=1.0,
            production_cache=cache, min_free_gib=args.min_free_gib, seed_base=protocol["seed_base"],
            require_production_cache=True, joint_activation=True, formats_by_qname=plan,
            checkpoint_dir=out / "checkpoints", model_identity=model_identity,
            checkpoint_identity_extra={"qualification_protocol_sha256": args.protocol_sha256})
        payload = compute_aura_cost_streamed(runner, calibration, formats, **kwargs)
        if payload["provenance"]["probe_identity_sha256"] != expected_probe_sha256:
            raise ValueError("streamed producer probe identity differs from frozen inputs")
        for name, rows in payload["costs"].items():
            for fmt, row in rows.items():
                operator = row["joint_operator_identity"]
                if identity_sha256(operator) != expected[name][fmt]:
                    raise ValueError("streamed operator differs from independently recorded production render")
        restored = compute_aura_cost_streamed(runner, calibration, formats, resume=True, **kwargs)
        if restored["costs"] != payload["costs"]:
            raise RuntimeError("checkpoint reload changed signed currency rows")
        identity["streamed_prefetch"] = runner.context.prefetch_summary()
        identity["checkpoint_reload_equal"] = True
    finally:
        runner.shutdown()
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    with (out / "joint.pkl").open("wb") as stream:
        pickle.dump(restored, stream, protocol=pickle.HIGHEST_PROTOCOL)
    payload, candidates, candidate_receipt = load_candidate_payload(out / "joint.pkl", plan, expected,
                                            expected_probe_sha256)
    dump(out / "candidate-handoff.json", candidate_receipt)
    scores, probe_pairs = compare_assignments(payload, candidates, protocol["assignments"])
    n_params = sum(payload["stats"][name]["n_params"] for name in plan)
    for key, score in scores.items():
        score["serialized_blob_bytes"] = sum(renders[name][fmt]["blob_bytes"]
                                              for name, fmt in score["assignment"].items())
        score["selected_quantizable_bpp"] = score["serialized_blob_bytes"] * 8 / n_params
        score["allocator_tensor_payload_bytes"] = sum(next(c.memory_bytes for c in candidates[name]
                                                            if c.fmt == fmt)
                                                      for name, fmt in score["assignment"].items())
    dump(out / "scores.json", scores)
    dump(out / "probe-contrasts.json", probe_pairs)
    model = resident_model()
    errors = _direct_oracle(model, calibration, cache, plan, payload)
    dump(out / "oracle.json", errors)
    if any(row["absolute_error"] > row["tolerance"] for row in errors):
        raise RuntimeError("real streamed producer disagrees with direct residual oracle")
    evaluate_splits(model, cache, protocol, out)
    identity.update(elapsed_seconds=time.time() - start, source_weights_restored=True,
                    completed=True, quantizable_parameters=n_params)
    dump(out / "identity.json", identity)
    dump(out / "receipt.json", {"completed": True,
        "artifacts": {str(path.relative_to(out)): file_sha256(path) for path in sorted(out.rglob("*"))
                      if path.is_file() and path.name != "receipt.json" and "streamed-offload" not in path.parts}})
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
