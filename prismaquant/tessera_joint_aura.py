"""Research joint AURA over exact, completed Tessera campaign anchors.

The campaign's scalar MSE/interpolation is evidence of which wires were made,
never a joint price. Original decoded renders enter ProductionWeightCache;
Tessera's existing source/H/settings receipt and decoder qualify them before
ordinary streamed joint AURA consumes the exact per-Linear candidate roster.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

from .cost_stage_checkpoint import (
    MANIFEST_SCHEMA, _load_unit, atomic_write_bytes, canonical_json_sha256, unit_path,
)

SCHEMA = "prismaquant.tessera_joint_aura.plan.v1"
PREPARED_SCHEMA = "prismaquant.tessera_joint_aura.prepared.v1"
CAMPAIGN_SCHEMA = "prismaquant.tessera_campaign_cost.v1"
CURRENCY = "output_mse_under_route_activation_contract"
STAGE = "Tessera campaign"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _bound(record, label):
    _require(isinstance(record, dict) and set(record) == {"path", "sha256"},
             f"{label}: independently bound path/SHA256 required")
    path = Path(record["path"])
    _require(_sha(path) == record["sha256"], f"{label}: artifact checksum changed")
    return path


def _same(actual, expected, label):
    _require(actual == expected, f"{label}: identity mismatch")


def _json(path, value):
    atomic_write_bytes(Path(path), (json.dumps(value, indent=2, sort_keys=True,
                                              allow_nan=False) + "\n").encode())


@dataclass
class MeasuredAnchorInput:
    inputs: dict
    payload: dict
    manifest: dict
    census: dict
    campaign_plan: dict
    cells: dict
    formats_by_qname: dict

    @property
    def total_render_bytes(self):
        return sum(2 * math.prod(self.census["unit_shapes"][name]) for name, _ in self.cells)

    def layer_render_bytes(self, layer_for_name):
        sizes = defaultdict(int)
        for name, _fmt in self.cells:
            sizes[layer_for_name(name)] += 2 * math.prod(self.census["unit_shapes"][name])
        return dict(sizes)


def load_measured_anchor_input(inputs, *, file_hash_workers=1):
    """Read a complete merged journal and select only its measured wire cells.

    This is a hash/receipt intake. Tensor/source/encoder verification occurs in
    ``prepare_cache`` using actual source weights and the original capture.
    Interpolated menu rows are deliberately excluded rather than converted.
    """
    from .production_weight_cache import _cache_weight_filename
    from tools.dispatch_tessera_campaign import _require_receipts

    _require(type(file_hash_workers) is int and file_hash_workers > 0,
             "positive file_hash_workers required")
    paths = {key: _bound(inputs[key], key) for key in (
        "campaign_plan", "census", "campaign_receipts", "merged_cost", "merged_checkpoint")}
    census = json.loads(paths["census"].read_text())
    plan = json.loads(paths["campaign_plan"].read_text())
    _same(plan.get("schema"), "prismaquant.tessera_campaign_plan.v1", "campaign plan schema")
    _same(Path(plan["census"]).resolve(), paths["census"].resolve(), "campaign census path")
    _same(paths["campaign_receipts"].resolve(),
          (paths["campaign_plan"].parent / "receipts.json").resolve(), "campaign receipt path")
    rows = plan["rows"]
    _require(len({row["row_id"] for row in rows}) == len(rows), "duplicate campaign row")
    _require_receipts(paths["campaign_plan"].parent, len(rows))
    owners, groups = {}, {}
    for row in rows:
        for name in row["members"]:
            _require(name not in owners, f"duplicate campaign unit {name}")
            owners[name] = Path(row["dir"])
        for group in row["groups"]:
            _require(group not in groups, f"duplicate campaign group {group}")
            groups[group] = row["row_id"]
    names = set(census["unit_shapes"])
    _same(set(owners), names, "complete census roster")
    _same(set(groups), set(census["anchor_groups"]), "complete census groups")
    _same(len(names), inputs["required_source_units"], "declared full source unit count")
    _same(len(groups), inputs["required_campaign_groups"], "declared full campaign group count")
    for group, members in census["anchor_groups"].items():
        owner = next(row for row in rows if row["row_id"] == groups[group])
        _require(set(members) <= set(owner["members"]), f"campaign group membership changed: {group}")

    payload = pickle.loads(paths["merged_cost"].read_bytes())
    _same(payload.get("schema"), CAMPAIGN_SCHEMA, "campaign cost schema")
    _same(payload.get("currency"), CURRENCY, "campaign scalar currency")
    _same(set(payload["costs"]), names, "complete merged cost roster")
    provenance = payload["provenance"]
    _same(provenance.get("cost_mode"), "production-render-score", "campaign cost mode")
    _same(provenance.get("model"), census["model"], "campaign model")
    _same(plan["model"], census["model"], "planned model")
    _require(provenance.get("stopped_early") is False, "campaign stopped before completing anchors")
    _same(provenance.get("campaign_fanout", {}).get("rows"),
          {row["row_id"]: sorted(row["groups"]) for row in rows}, "complete merged fanout")

    manifest = json.loads(paths["merged_checkpoint"].read_text())
    _same(manifest.get("schema"), MANIFEST_SCHEMA, "campaign checkpoint schema")
    _same(manifest.get("stage"), STAGE, "campaign checkpoint stage")
    identity = manifest["identity"]
    seal = canonical_json_sha256(identity, where="joint anchor input")
    _same(seal, manifest.get("identity_sha256"), "campaign checkpoint seal")
    _same(identity.get("campaign_schema"), CAMPAIGN_SCHEMA, "checkpoint campaign schema")
    _same(identity.get("currency"), CURRENCY, "checkpoint scalar currency")
    _same(set(identity["units"]), names, "complete checkpoint identity roster")
    listed = [row["qname"] for row in manifest["units"]]
    _require(len(listed) == len(names) and set(listed) == names, "incomplete checkpoint unit roster")
    for key in ("prismaquant_source_sha256", "encoder_source_sha256"):
        value = identity.get(key)
        _require(isinstance(value, str) and len(value) == 64 and
                 all(c in "0123456789abcdef" for c in value), f"missing checkpoint {key}")
    parts = paths["merged_checkpoint"].with_name(paths["merged_checkpoint"].name + ".parts")
    for row in manifest["units"]:
        _same(parts / row["file"], unit_path(parts, row["qname"]), "canonical checkpoint unit path")

    cells, formats = {}, {}
    wire_dir = Path(provenance["wire_dir"])
    for name in sorted(names):
        state = _load_unit(unit_path(parts, name), stage=STAGE, qname=name, identity_sha256=seal)
        _require(isinstance(state, dict) and set(state) - {"unservable"} == {"anchors", "wire_records"},
                 f"{name}: incomplete measured anchor journal")
        anchors = {anchor["format_name"]: anchor for anchor in state["anchors"]}
        _require(anchors and len(anchors) == len(state["anchors"]) and
                 set(anchors) == set(state["wire_records"]), f"{name}: anchor/receipt coverage differs")
        measured = {fmt for fmt, row in payload["costs"][name].items()
                    if row.get("output_mse_measured") is True}
        _same(set(anchors), measured, f"{name}: measured payload/journal coverage")
        unit = identity["units"][name]
        _same(unit["weight"]["shape"], census["unit_shapes"][name], f"{name}: census source shape")
        for fmt, anchor in sorted(anchors.items()):
            row = payload["costs"][name][fmt]
            _require(fmt in unit["menu"] and anchor["qname"] == name, f"{name}: anchor outside exact menu")
            _require(row.get("cost_source") == "tessera_campaign_measured" and
                     row.get("tessera_provenance") == "measured" and row.get("currency") == CURRENCY,
                     f"{name}@{fmt}: interpolated or foreign measured row")
            for target, source in (("output_mse", "dloss"), ("tessera_family", "family"),
                    ("tessera_body_rate_q256", "body_rate_q256"), ("activation_contract", "activation_contract"),
                    ("activation_quantized", "activation_quantized"), ("wire_bytes", "wire_bytes"),
                    ("input_global_scale", "input_global_scale")):
                _same(row.get(target), anchor.get(source), f"{name}@{fmt}: measured {target}")
            _require(type(anchor["dloss"]) in (int, float) and math.isfinite(anchor["dloss"])
                     and anchor["dloss"] >= 0, f"{name}@{fmt}: invalid measured value")
            _same(row["hessian_identity"].get("applied"), anchor["hessian_applied"], f"{name}: H applicability")
            for key in ("supplied", "capture_sha256", "text_sha256", "fit_ids_sha256", "fit_tokens"):
                _same(row["hessian_identity"].get(key), provenance["hessian"].get(key), f"{name}: measured H {key}")
            if anchor.get("input_global_scale") is not None:
                _same(anchor["input_global_scale"], unit.get("input_global_scale"), f"{name}: checkpoint scale")
                _same(anchor["input_global_scale"], provenance["activation_static_scales"]["units"].get(name),
                      f"{name}: merged static scale")
            record = state["wire_records"][fmt]
            recorded = record["identity"]
            _same(recorded.get("unit"), name, f"{name}: wire unit")
            _same(recorded.get("source"), unit["weight"], f"{name}: recorded source")
            _same(recorded.get("encoder_source_sha256"), identity["encoder_source_sha256"], f"{name}: encoder source")
            _same(recorded["recipe"].get("q256"), anchor["body_rate_q256"], f"{name}: wire rung")
            if anchor["hessian_applied"]:
                _same(recorded["calibration"]["hessian"], unit["hessian"], f"{name}: recorded H")
            else:
                _same(recorded.get("calibration"), None, f"{name}: unexpected recorded H")
            filename = record["file"]
            _require(isinstance(filename, str) and Path(filename).name == filename and
                     filename not in {".", ".."}, f"{name}: escaping wire filename")
            wire = wire_dir / filename
            _require(not wire.is_symlink() and wire.resolve().parent == wire_dir.resolve(), f"{name}: escaping wire path")
            _same(wire.stat().st_size, record["blob_bytes"], f"{name}: wire size")
            render = owners[name] / "cache" / _cache_weight_filename(name, fmt)
            _require(render.is_file(), f"{name}@{fmt}: original decoded PWC shard missing")
            cells[name, fmt] = {"anchor": anchor, "record": record, "wire": str(wire.resolve()),
                               "render": str(render.resolve())}
        formats[name] = (*sorted(anchors), "BF16")
    def verify_files(item):
        pair, cell = item
        wire, render = Path(cell["wire"]), Path(cell["render"])
        # Metadata is only a race detector around the actual content hash.
        # Every byte is still hashed; neither timestamps nor a previous run
        # authorize reuse. Existing per-consumption render checks remain below.
        def signature(path):
            stat = path.stat()
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        before = [signature(p) for p in (wire, render)]
        _same(_sha(wire), cell["record"]["blob_sha256"], f"{pair}: wire checksum")
        digest = _sha(render)
        after = [signature(p) for p in (wire, render)]
        _same(after, before, f"{pair}: input files changed while hashing")
        return pair, digest

    if file_hash_workers == 1:
        verified_files = map(verify_files, cells.items())
        for pair, digest in verified_files:
            cells[pair]["render_file_sha256"] = digest
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=file_hash_workers, thread_name_prefix="anchor-file-hash") as workers:
            for pair, digest in workers.map(verify_files, cells.items()):
                cells[pair]["render_file_sha256"] = digest
    return MeasuredAnchorInput(dict(inputs), payload, manifest, census, plan, cells, formats)


def calibrated_maxima(data, profile):
    """Reuse the producer's full-census fused scale policy; never invert G."""
    from . import tessera_campaign as tc
    from .nvfp4_activation_contract import unify_fused_sibling_max_abs

    positive = {name: float(value) for name, value in data.census["max_abs"].items()
                if float(value) > 0.0}
    maxima = unify_fused_sibling_max_abs(positive, profile=profile, tolerate_profile_errors=True)
    scales, policy = tc._static_input_scales(data.census["max_abs"], profile=profile)
    stamped = data.payload["provenance"]["activation_static_scales"]
    _same(policy, stamped["policy"], "campaign static scale policy")
    _same(scales, stamped["units"], "campaign fused static scales")
    return maxima, scales


def verify_anchor_render(cell, source_weight, rendered_weight, *, calibration_source,
                         projected_unit, static_scales, bound_unit=None, reader=None):
    """Re-derive encoder inputs from actual source/H and compare decoded bytes."""
    import torch
    from tessera.unit_artifact import read_unit_artifact
    from . import tessera_campaign as tc
    from .production_weight_cache import _cb_cache_tensor_identity

    anchor = tc.CampaignAnchor(**cell["anchor"])
    name, fmt = anchor.qname, anchor.format_name
    _require(source_weight.dtype == rendered_weight.dtype == torch.bfloat16 and
             source_weight.ndim == 2 and rendered_weight.shape == source_weight.shape,
             f"{name}@{fmt}: source/render BF16 shape differs")
    source_receipt = (None if bound_unit is None else bound_unit.source_receipt(source_weight))
    _require((bound_unit is not None or bool(torch.isfinite(source_weight).all())) and
             bool(torch.isfinite(rendered_weight).all()), f"{name}@{fmt}: source/render is nonfinite")
    expected = tc._checkpoint_anchor_identity(anchor,
        weights={name: source_weight}, menus={name: [SimpleNamespace(format_name=fmt)]},
        calibration_source=calibration_source, static_scales=static_scales,
        projected_units={} if projected_unit is None else {name: projected_unit},
        **({} if bound_unit is None else {"bound_unit": bound_unit}))
    blob = Path(cell["wire"]).read_bytes()
    verifier = tc._checkpoint_identity_api() if reader is None else reader
    verifier.verify_cached_unit(blob, cell["record"], expected)
    decode = read_unit_artifact if reader is None else reader.read_unit_artifact
    decoded = decode(blob, device=str(rendered_weight.device)).to(torch.bfloat16)
    _require(torch.equal(decoded, rendered_weight), f"{name}@{fmt}: decoded wire differs from original PWC render")
    del decoded
    return {"source_weight": (_cb_cache_tensor_identity(source_weight)
                              if source_receipt is None else source_receipt),
            "rendered_weight": _cb_cache_tensor_identity(rendered_weight),
            "encoding_identity_sha256": canonical_json_sha256(expected, where="joint anchor encoding"),
            "wire_sha256": hashlib.sha256(blob).hexdigest(),
            "render_file_sha256": cell["render_file_sha256"]}


def _live_targets(runner, names):
    from .aura_cost import _target_linears
    from .routed_experts import profile_declared_packed_expert_projections

    targets = _target_linears(runner.model, include_routed_experts=True, profile=runner.profile)
    packed = profile_declared_packed_expert_projections(runner.model, runner.profile)
    targets.update({member.qname: member for member in packed})
    _require(set(names) <= set(targets), "census units are absent from the actual streamed source")
    return {name: targets[name] for name in names}


def prepare_cache(runner, data, *, capture, max_render_bytes, reader=None):
    """Qualify original per-layer inputs and return the existing PWC object.

    Only the original calibration/PWC/source prefetch mechanisms own tensors.
    PWC's LRU records absolute donor paths so compact/release stays reversible
    even though the merged renders have more than one original directory.
    """
    import torch
    from . import tessera_calibration_cache as cc, tessera_hessian as th, tessera_campaign as tc
    from .joint_aura import activation_identity, prefetch_joint_cache
    from .production_weight_cache import ProductionWeightCache
    from .routed_experts import PackedExpertProjection, refresh_packed_expert_projections
    from . import format_registry as fr

    _require(type(max_render_bytes) is int and max_render_bytes > 0, "positive PWC residency budget required")
    capture_path = _bound(capture, "canonical capture")
    stamped_capture = data.payload["provenance"].get("calibration_cache")
    _same(capture, stamped_capture, "priced canonical capture")
    manifest = cc.require_capture_contract(capture_path, expected_sha256=capture["sha256"])
    recorded = manifest["identity"]
    # This verifies recorded canonical capture provenance plus current source
    # bytes/runtime. It does not pretend a from-config streaming skeleton is an
    # ordinary from_pretrained model or manufacture an initialization witness.
    expected = cc.capture_identity(data.inputs["census"]["path"],
        calibration=data.payload["provenance"]["hessian"]["calibration_identity"],
        max_act_rows=recorded["max_act_rows"],
        model_load_contract=data.census["model_load_contract"],
        attention_implementation=data.census["attention_implementation"])
    _same(expected, recorded, "current source/canonical capture")
    _same(data.manifest["identity"]["calibration"], recorded["calibration"], "journal/canonical draw")
    maxima, scales = calibrated_maxima(data, runner.profile)
    cache = ProductionWeightCache(
        weights={pair: cell["render"] for pair, cell in data.cells.items()},
        levers={"tessera_campaign": True}, activation_max_abs=maxima,
        metadata={"schema": PREPARED_SCHEMA, "inputs": data.inputs,
                  "reader_identity": None if reader is None else reader.identity})
    cache.enable_lru(max_render_bytes)
    targets = _live_targets(runner, data.formats_by_qname)
    layers = defaultdict(list)
    for name in targets:
        layers[runner.layer_index_for_qname(name)].append(name)
    projected = {name: unit for units in (data.census.get("expert_projection") or {}).get("stacks", {}).values()
                 for name, unit in units.items()}
    renders = {name: tuple(fmt for fmt in fmts if fmt != "BF16")
               for name, fmts in data.formats_by_qname.items()}
    verified, telemetry = {}, []
    for depth in range(min(runner.num_layers, runner.prefetch_lookahead + 1)):
        runner.context.schedule_prefetch(depth)
    for layer in range(runner.num_layers):
        names = sorted(layers.get(layer, ()))
        runner.context.install(layer, require_prefetched=runner.require_prefetched_residency)
        runner.context.schedule_prefetch(layer + runner.prefetch_lookahead)
        members = [targets[name] for name in names if isinstance(targets[name], PackedExpertProjection)]
        try:
            targets.update({member.qname: member for member in refresh_packed_expert_projections(members, runner.profile)})
            if not names:
                continue
            (acts, hessians, _counts, _maxima), _receipt = cc.prefetch_capture(capture_path,
                expected_sha256=capture["sha256"], expected_identity=expected,
                census=data.census, names=names, device=runner.device)
            calibration_source = th.activation_source(hessians, expected["calibration"])
            stats = prefetch_joint_cache(cache, names, renders, max_resident_bytes=max_render_bytes)
            for name in names:
                source_weight = targets[name].weight.detach()
                anchors = [tc.CampaignAnchor(**data.cells[name, fmt]["anchor"]) for fmt in renders[name]]
                with tc.bind_checkpoint_unit_identity(anchors, source_weight=source_weight,
                        calibration_source=calibration_source, projected_unit=projected.get(name),
                        static_scales=scales) as bound_unit:
                    for fmt in renders[name]:
                        cell = data.cells[name, fmt]
                        _same(_sha(cell["render"]), cell["render_file_sha256"], f"{name}: original render file changed")
                        rendered = cache.get(name, fmt).to(runner.device)
                        record = verify_anchor_render(cell, source_weight, rendered,
                            calibration_source=calibration_source,
                            projected_unit=projected.get(name), static_scales=scales,
                            bound_unit=bound_unit, reader=reader)
                        activation = activation_identity(fr.get_format(fmt), cache.activation_max_abs, name)
                        _same(activation["input_global_scale"], cell["anchor"].get("input_global_scale"),
                              f"{name}@{fmt}: joint/campaign static scale")
                        record["activation"] = activation
                        verified[name, fmt] = record
                        del rendered
            telemetry.append({"layer": layer, **stats})
            print(json.dumps({"qualified_layer": layer, "qualified_cells": len(verified),
                              "total_cells": len(data.cells), "prefetch": stats}), flush=True)
            del acts, hessians, calibration_source, source_weight
        finally:
            cache.compact_for_pickle()
            runner.context.unload(layer)
            targets.update({member.qname: member for member in refresh_packed_expert_projections(members, runner.profile)})
    _same(set(verified), set(data.cells), "complete qualified wire/render roster")
    cache.metadata.update({"verified_cells": verified, "prefetch": telemetry})
    return cache


def _source_prefetch(config):
    prefetch = config.get("source_prefetch")
    fields = {"max_cache_slots", "prefetch_workers", "prefetch_lookahead",
              "cache_headroom_gb", "prefetch_min_available_gb",
              "require_prefetched_residency"}
    _require(isinstance(prefetch, dict) and set(prefetch) == fields,
             "explicit complete source_prefetch settings required")
    _require(prefetch["require_prefetched_residency"] is True,
             "source_prefetch must require prefetched residency")
    for name in ("max_cache_slots", "prefetch_workers", "prefetch_lookahead"):
        _require(type(prefetch[name]) is int and prefetch[name] > 0,
                 f"source_prefetch requires positive {name}")
    _require(prefetch["prefetch_lookahead"] < prefetch["max_cache_slots"],
             "source_prefetch lookahead must fit the declared cache slots")
    for name in ("cache_headroom_gb", "prefetch_min_available_gb"):
        _require(type(prefetch[name]) in (int, float) and
                 math.isfinite(prefetch[name]) and prefetch[name] > 0,
                 f"source_prefetch requires positive finite {name}")
    return dict(prefetch)


def _load_plan(path, digest):
    path = _bound({"path": str(path), "sha256": digest}, "joint anchor plan")
    config = json.loads(path.read_text())
    _same(config.get("schema"), SCHEMA, "joint anchor plan schema")
    _source_prefetch(config)
    execution = config["execution"]
    _require(type(config.get("file_hash_workers", 1)) is int and config.get("file_hash_workers", 1) > 0,
             "positive file_hash_workers required")
    for name, minimum in (("n_calib_samples", 1), ("calib_seqlen", 1),
                          ("probe_microbatch", 1), ("n_probes", 2)):
        _require(type(execution.get(name)) is int and execution[name] >= minimum,
                 f"explicit positive {name} required")
    _require(type(execution.get("seed_base")) is int, "explicit probe seed required")
    _same(execution.get("token_scope"), "all", "full-draw joint token scope")
    _same(execution.get("temperature"), 1.0, "joint probe temperature")
    _same(execution.get("production_act_scales"), "0", "campaign optional activation clipping")
    _same(config.get("profile_tool"), "cprofile", "full-duration in-process profiler")
    for name in ("max_render_bytes", "max_gpu_bytes"):
        _require(type(config.get(name)) is int and config[name] > 0, f"positive {name} required")
    _require(type(config.get("min_free_gib")) in (int, float) and config["min_free_gib"] >= 0,
             "nonnegative memory floor required")
    return config


def _io_counters():
    values = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value)
    return values


def execute(command, config, *, plan_sha256, prepared=None, resume=False):
    """Execute one admitted preparation or one dependent cost action."""
    import cProfile
    import io
    import os
    import pstats
    import socket
    import torch
    from .aura_cost import compute_aura_cost_streamed, _aura_source_sha256
    from .calibration_data import load_calibration_input
    from .cost_streaming import build_streamed_causal_lm, build_streamed_model_identity
    from .joint_aura import source_execution_identity, validate_joint_aura_entry
    from .model_profiles import detect_profile
    from .production_weight_cache import ProductionWeightCache
    from .gpu_guard import require_cuda_hot_path
    from .tessera_reader import load_declared_reader

    require_cuda_hot_path("tessera_joint_aura", "cuda")
    os.environ["PRISMAQUANT_PROD_ACT_SCALES"] = config["execution"]["production_act_scales"]
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    execution = config["execution"]
    root = Path(config["output_root"]) / command
    root.mkdir(parents=True, exist_ok=True)
    result = {"schema": "prismaquant.tessera_joint_aura.execution.v1", "command": command,
              "plan_sha256": plan_sha256, "env": {"host": socket.gethostname(),
                  "started_epoch": time.time(), "torch": str(torch.__version__),
                  "cuda": torch.version.cuda, "affinity": sorted(os.sched_getaffinity(0))},
              "phases": [], "passed": False}
    profiler = cProfile.Profile()
    runner = None
    completion_path = completion = output = payload = None
    started, before_io = time.time(), _io_counters()
    profiler.enable()
    try:
        file_hash_workers = config.get("file_hash_workers", 1)
        _require(type(file_hash_workers) is int and 0 < file_hash_workers <= len(os.sched_getaffinity(0)),
                 "file_hash_workers exceeds PB-assigned CPU affinity")
        data = load_measured_anchor_input(config["inputs"],
            **({} if file_hash_workers == 1 else {"file_hash_workers": file_hash_workers}))
        result["file_hash_workers"] = file_hash_workers
        reader = load_declared_reader(config.get("reader"))
        reader_identity = None if reader is None else reader.identity
        result["reader_identity"] = reader_identity
        _same(config["model"], data.census["model"], "requested source model")
        _same(data.census["attention_implementation"], "eager", "qualified source attention")
        ids, calibration = load_calibration_input(config["calibration_input"]["path"],
            expected_sha256=config["calibration_input"]["sha256"],
            n_samples=execution["n_calib_samples"], seqlen=execution["calib_seqlen"])
        original_draw = data.payload["provenance"]["hessian"]["calibration_identity"]
        for name in ("fit_ids_sha256", "text_sha256", "nsamples", "seqlen", "seed"):
            _same(calibration["provenance"].get(name), original_draw.get(name), f"original full draw {name}")
        result["calibration_input"] = calibration
        source_prefetch = _source_prefetch(config)
        runner = build_streamed_causal_lm(config["model"], device=torch.device("cuda"),
            dtype=torch.bfloat16, offload_folder=str(root / "offload"),
            profile=detect_profile(config["model"]), attn_implementation="eager",
            **source_prefetch)
        result["source_prefetch"] = source_prefetch
        source = build_streamed_model_identity(runner, config["model"],
                                               identity_cache_path=root / "source-identity.json")
        source_execution = source_execution_identity(runner.model)
        layer_bytes = data.layer_render_bytes(runner.layer_index_for_qname)
        _require(max(layer_bytes.values()) <= config["max_render_bytes"],
                 "largest measured candidate layer exceeds explicit PWC budget")
        result.update(source_model_identity=source, source_execution=source_execution,
                      units=len(data.formats_by_qname), measured_cells=len(data.cells),
                      layer_render_bytes=layer_bytes)
        implementation = _aura_source_sha256()
        if command == "prepare":
            _require(prepared is None and not resume, "preparation does not consume a prepared cache or cost resume")
            completion_path = root / "prepared.json"
            _require(not completion_path.exists(), "prepared completion already exists; use its bound record")
            cache = prepare_cache(runner, data, capture=config["canonical_capture"],
                                  max_render_bytes=config["max_render_bytes"], reader=reader)
            cache.metadata.update(plan_sha256=plan_sha256, source_model_identity=source,
                                  source_execution=source_execution, implementation_sha256=implementation)
            cache.compact_for_pickle()
            cache_path = root / "production.pkl"
            atomic_write_bytes(cache_path, pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))
            completion = {"schema": PREPARED_SCHEMA, "status": "complete", "plan_sha256": plan_sha256,
                "implementation_sha256": implementation, "source_model_identity": source,
                "reader_identity": reader_identity,
                "source_execution": source_execution, "calibration_input": calibration,
                "production_cache": {"path": str(cache_path), "sha256": _sha(cache_path)},
                "formats_by_qname": data.formats_by_qname, "measured_cells": len(data.cells)}
        else:
            _require(prepared is not None, "cost execution requires independently bound prepared inputs")
            completion = json.loads(_bound(prepared, "prepared anchors").read_text())
            _same(completion.get("schema"), PREPARED_SCHEMA, "prepared schema")
            _same(completion.get("status"), "complete", "prepared completion")
            for key, value in (("plan_sha256", plan_sha256), ("implementation_sha256", implementation),
                               ("source_model_identity", source), ("source_execution", source_execution),
                               ("calibration_input", calibration), ("measured_cells", len(data.cells)),
                               ("reader_identity", reader_identity)):
                _same(completion.get(key), value, f"prepared {key}")
            _same(completion["formats_by_qname"], {n: list(v) for n, v in data.formats_by_qname.items()},
                  "prepared exact candidate roster")
            cache = pickle.loads(_bound(completion["production_cache"], "qualified PWC").read_bytes())
            _require(isinstance(cache, ProductionWeightCache), "prepared cache is not ProductionWeightCache")
            _same(cache.metadata["inputs"], data.inputs, "prepared source bindings")
            _same(cache.metadata.get("reader_identity"), reader_identity, "prepared reader identity")
            _same(set(cache.metadata["verified_cells"]), set(data.cells), "prepared verified cell coverage")
            _same(cache.weights, {pair: cell["render"] for pair, cell in data.cells.items()}, "prepared original render paths")
            for pair, cell in data.cells.items():
                _same(cache.metadata["verified_cells"][pair]["render_file_sha256"], cell["render_file_sha256"],
                      f"{pair}: qualified render changed")
                _same(cache.metadata["verified_cells"][pair]["wire_sha256"], cell["record"]["blob_sha256"],
                      f"{pair}: qualified wire changed")
            _live_targets(runner, data.formats_by_qname)
            formats = list(dict.fromkeys(fmt for values in data.formats_by_qname.values() for fmt in values))
            payload = compute_aura_cost_streamed(runner, ids.to(runner.device), formats,
                n_probes=execution["n_probes"], probe_microbatch=execution["probe_microbatch"],
                seed_base=execution["seed_base"], token_scope="all", temperature=1.0,
                production_cache=cache, require_production_cache=True, joint_activation=True,
                include_routed_experts=True, include_lm_head=False, dw_dtype="float32",
                min_free_gib=config["min_free_gib"], formats_by_qname=data.formats_by_qname,
                checkpoint_dir=Path(config["output_root"]) / "checkpoints", resume=resume,
                model_identity=source, profile=runner.profile,
                checkpoint_identity_extra={"tessera_joint_anchor_plan_sha256": plan_sha256,
                    "prepared_anchor_sha256": prepared["sha256"], "calibration_input": calibration,
                    "reader_identity": reader_identity})
            _same(set(payload["costs"]), set(data.formats_by_qname), "complete joint output roster")
            for name, rows in payload["costs"].items():
                _same(set(rows), set(data.formats_by_qname[name]), f"{name}: joint output candidates")
                for row in rows.values():
                    _require(validate_joint_aura_entry(row), f"{name}: invalid measured joint cost")
            payload["provenance"]["tessera_joint_anchors"] = {
                "plan_sha256": plan_sha256, "prepared": prepared, "inputs": data.inputs,
                "calibration_input": calibration, "measured_cells": len(data.cells)}
            output = root / "joint-cost.pkl"
        torch.cuda.synchronize()
        result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        _require(result["peak_gpu_bytes"] <= config["max_gpu_bytes"], "observed GPU allocation exceeds declared budget")
        # A completion is published only after the source residency owner has
        # shut down successfully as well as after the allocation gate passes.
        completed_runner, runner = runner, None
        completed_runner.shutdown()
        if command == "prepare":
            _json(completion_path, completion)
            result["prepared"] = {"path": str(completion_path), "sha256": _sha(completion_path)}
        else:
            atomic_write_bytes(output, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            result["cost"] = {"path": str(output), "sha256": _sha(output)}
        result["passed"] = True
    finally:
        profiler.disable()
        profiler.dump_stats(str(root / "profile.pstats"))
        text = io.StringIO()
        pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(100)
        (root / "profile.txt").write_text(text.getvalue())
        result["env"]["finished_epoch"] = time.time()
        result["phases"].append({"phase": command, "kind": "profile", "start_epoch": started,
                                 "end_epoch": result["env"]["finished_epoch"]})
        result["io_before"], result["io_after"] = before_io, _io_counters()
        _json(root / "results.json", result)
        if runner is not None:
            runner.shutdown()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.prepared) != bool(args.prepared_sha256):
        parser.error("--prepared and --prepared-sha256 are required together")
    config = _load_plan(args.plan, args.plan_sha256)
    result = execute(args.command, config, plan_sha256=args.plan_sha256,
        prepared=None if args.prepared is None else {"path": str(args.prepared), "sha256": args.prepared_sha256},
        resume=args.resume)
    print(json.dumps({key: result[key] for key in ("command", "passed", "units", "measured_cells")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
