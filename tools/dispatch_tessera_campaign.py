#!/usr/bin/env python3
"""Fan a Tessera anchor campaign out across the fleet, one row per anchor group.

``prismaquant.tessera_campaign`` prices a rate surface per (unit, family), and
every one of those surfaces is independent work: its anchors, its
leave-one-out gate and its refusal are its own.  Run as one process it is a
single exclusive GPU action, and the second box sits idle beside it
(RobTand/prismaquant#282).  This lays the same campaign out as N PrismaBuild
rows and puts the pieces back together.

The quantum is the **fused anchor group**, not the bare unit.  Anchors are
placed per group: the group shares one rung grid, and the group's worst member
drives every split (``tessera_campaign`` round loop).  A group is therefore the
smallest scope whose measured values do not depend on what else the run priced,
which is what makes the merged table equal to the monolith's rather than merely
similar to it.

Four steps, and each one is separately re-runnable:

``census``
    One cheap GPU row: a calibration forward over the whole scope that counts
    each unit's rows and reports the anchor grouping.  The counts are what let
    every later row stamp the **scope's** ``fit_tokens`` rather than its own
    selection's, so the sharded table carries one Hessian identity -- the same
    one a whole-scope run carries.  The grouping is what lets ``plan`` lay out
    rows without loading the model.

``plan``
    One ``--units`` selection file per row and one pbcampaign manifest.  Rows
    are portable (no host pin), not exclusive, GPU-demanding, and carry a
    memory demand derived from the checkpoint's size and the selection's
    shapes.

``submit``
    ``pbcampaign`` over that manifest.  Re-running it **is** the resume: a
    finished row is a CAS hit that runs nothing and a running row is attached
    to by its job id, so there is no second dispatcher here deciding what to
    skip.

``merge``
    One ``cost.pkl``, one ``cost.anchors.json`` and one export-inputs cache
    from the rows, refusing on any identity the rows do not already share.  The
    merged Hessian capture is the union of the rows' -- the same H under the
    same counts and the same provenance a whole-scope run writes -- so its
    digest is recomputed rather than asserted, and every row's
    ``capture_sha256`` is re-stamped to it.

Seeding from a campaign already in flight
-----------------------------------------
``--seed-checkpoint`` on a planned row hands the monolith's stored anchors to
that row's own gates: the producer input identity is recomputed from the row's
weights, menu, Hessian and static scale, and the cached wire is re-verified
against it, so a seeded row adopts only bytes it would itself have encoded.
The adaptive state needs nothing else -- ``grid``, the leave-one-out error and
the stop reason are all recomputed from the anchor set at the top of every
round -- so adopting the anchors resumes the group exactly where it stood.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

PBCAMPAIGN = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbcampaign.py")

#: What ``plan`` writes beside the manifest, so ``merge`` reads the row layout
#: from the plan rather than from the directory listing it happens to find.
PLAN_SCHEMA = "prismaquant.tessera_campaign_plan.v1"

#: Provenance fields every row must agree on before a merge is possible.  Each
#: one describes the run, not the selection, so a disagreement means two
#: campaigns are being merged into one table.
SHARED_PROVENANCE = (
    "menu_mode", "tp_degree", "model", "nsamples", "seqlen", "max_act_rows",
    "layer_stride", "anchors_round_one", "max_rounds", "anchor_budget",
    "loo_gate", "max_artifact_bpp", "cost_mode", "rate_band",
)

#: Hessian identity fields every row must already agree on.  ``capture_sha256``
#: is deliberately absent: it is the digest of the capture a row wrote for its
#: own units, and reconciling it is what the merge is for.
SHARED_HESSIAN = (
    "supplied", "text_sha", "token_count", "text_sha256", "fit_ids_sha256",
    "fit_tokens", "kwarg",
)


class MergeRefused(RuntimeError):
    """The rows do not describe one campaign."""


# ---------------------------------------------------------------------------
# The run spec
# ---------------------------------------------------------------------------

def load_spec(path: Path) -> dict:
    """Read the shared half of every row: model, campaign flags, fleet demand.

    Everything a row does *not* share -- its selection, its output paths -- is
    computed here, so the spec cannot accidentally pin two rows to one file.
    """
    spec = json.loads(Path(path).read_text())
    for field in ("model", "campaign_argv", "cwd", "python", "env"):
        if field not in spec:
            raise RuntimeError(f"{path}: spec has no {field!r}")
    forbidden = {"--model", "--out", "--cache-dir", "--checkpoint", "--units",
                 "--calibration-census", "--census-out", "--seed-checkpoint",
                 "--seed-wire-dir"}
    named = forbidden.intersection(spec["campaign_argv"])
    if named:
        raise RuntimeError(
            f"{path}: campaign_argv names {sorted(named)}, which this tool "
            "owns per row")
    if "--deadline-seconds" in spec["campaign_argv"]:
        # The in-process deadline stops a run mid-round, in the sorted-key
        # order the round's pending list happens to have; two rows stopped that
        # way price different anchor sets than one run would have. A row's
        # deadline is PrismaBuild's ``timeout_s`` and its retry, which restarts
        # the row against its own journal.
        raise RuntimeError(
            f"{path}: campaign_argv sets --deadline-seconds; a fanned-out row "
            "takes its deadline from the fleet, not from inside the round loop")
    return spec


def _model_bytes(model: str) -> int:
    root = Path(model)
    return sum(path.stat().st_size for path in root.glob("*.safetensors"))


def _row_memory_gb(spec: dict, members: list[str], census: dict) -> int:
    """The row's memory demand, from what the row actually holds.

    Three terms, each a measured quantity rather than a guess:

    * the checkpoint, which is loaded whole in ``bfloat16`` and is the same for
      every row;
    * the selection's Hessians, ``in x in`` in fp32 per member, which is the
      accumulator ``_collect_activations`` keeps;
    * the selection's retained scoring rows, ``max_act_rows x in`` in fp32.

    plus the spec's declared headroom for the forward pass and the encoder.
    """
    gib = 1024 ** 3
    shapes = census.get("unit_shapes") or {}
    hessian = sum(int(shapes.get(name, [0, 0])[1]) ** 2 * 4 for name in members)
    rows = sum(int(shapes.get(name, [0, 0])[1]) * int(spec.get("max_act_rows", 512)) * 4
               for name in members)
    total = _model_bytes(spec["model"]) + hessian + rows
    return int(math.ceil(total / gib)) + int(spec.get("headroom_gb", 24))


def require_rows_fit(mem_gb: "list[int]", per_box: int, budget) -> int:
    """Check that a box can hold ``per_box`` of these rows, and say so.

    Concurrency is a property of the row's demand, not a flag: PrismaBuild
    admits as many rows as a box's memory holds.  So this checks rather than
    sets -- shrinking a row's declared demand to force co-residency would be
    reserving less than the row holds.  Returns the widest demand.
    """
    if per_box < 1:
        raise RuntimeError("--rows-per-box must be at least 1")
    widest = max(mem_gb)
    if budget is not None and widest * per_box > int(budget):
        raise RuntimeError(
            f"--rows-per-box {per_box} does not fit: the widest row demands "
            f"{widest} GB and the spec declares a {int(budget)} GB box, so at "
            f"most {int(budget) // widest} of these rows are co-resident. "
            "Reduce --groups-per-row, or make the quantum hold less than the "
            "whole checkpoint.")
    print(f"[dispatch] widest row demands {widest} GB; "
          f"--rows-per-box {per_box} needs {widest * per_box} GB per box"
          + (f" (spec declares {int(budget)} GB)" if budget is not None else
             " (the spec declares no box budget, so this is unchecked)"))
    return widest


def _row(spec: dict, argv: list[str], *, mem_gb: int, timeout_s: int) -> dict:
    row = {
        "argv": [spec["python"], "-u", "-m", "prismaquant.tessera_campaign", *argv],
        "cwd": spec["cwd"],
        "demand": {"gpu": 1, "cpu": int(spec.get("cpus", 4)), "mem_gb": int(mem_gb)},
        "env": dict(spec["env"]),
        "tags": list(spec.get("tags", ["gb10"])),
        "timeout_s": int(timeout_s),
        # A row is one memoized action and a retry re-runs the same argv over
        # the same checkpoint, which is exactly what the journal is for.  The
        # policy is sealed into the action key, so it is spelled even though
        # pbcampaign submits every row detached and cannot retry one itself.
        "retry_safe": True,
    }
    return row


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------

def cmd_census(args) -> int:
    spec = load_spec(Path(args.spec))
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    census_path = workspace / "census.json"
    manifest = workspace / "census-manifest.json"
    row = _row(
        spec,
        ["--model", spec["model"],
         "--out", str(workspace / "census-unused.pkl"),
         "--cache-dir", str(workspace / "census-cache"),
         "--census-out", str(census_path),
         *spec["campaign_argv"]],
        mem_gb=_row_memory_gb(spec, [], {}),
        timeout_s=int(args.timeout_s),
    )
    manifest.write_text(json.dumps([row], indent=2) + "\n")
    print(f"[dispatch] census manifest {manifest}")
    if args.submit:
        return _pbcampaign(manifest, wait_s=args.wait_s,
                           receipts=workspace / "census-receipts.json")
    return 0


def _pbcampaign(manifest: Path, *, wait_s: int, receipts: Path | None = None) -> int:
    """Run the campaign and keep the fleet's own row table.

    The table is what says a row *ran*, as opposed to having been accepted:
    every row reports a key, the host it executed on and its exit status, and
    ``merge`` refuses without it.  A submission acknowledgement is not a result.
    """
    command = [sys.executable, str(PBCAMPAIGN), "--wait-s", str(wait_s), str(manifest)]
    print("[dispatch] " + " ".join(command), flush=True)
    completed = subprocess.run(command, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(completed.stdout, flush=True)
    if receipts is not None:
        rows = _parse_row_table(completed.stdout)
        receipts.write_text(json.dumps(
            {"manifest": str(manifest), "returncode": completed.returncode,
             "rows": rows}, indent=2) + "\n")
        print(f"[dispatch] {len(rows)} row receipts -> {receipts}")
    return completed.returncode


def _parse_row_table(text: str) -> list[dict]:
    """The ``key status transport job host elapsed rc receipt note`` table.

    Read by the header's own column offsets rather than by splitting on
    whitespace: ``pbwait`` left-justifies every cell to a common width, and a
    cell can hold a space -- ``rc`` renders ``1 (action 137)`` when the
    launcher's status and the action's differ, which is exactly the failing
    row a whitespace split would drop.
    """
    rows: list[dict] = []
    header: list[tuple[str, int, int]] | None = None
    for line in text.splitlines():
        if header is None:
            if line.split()[:2] != ["key", "status"]:
                continue
            names = line.split()
            starts = []
            cursor = 0
            for name in names:
                cursor = line.index(name, cursor)
                starts.append(cursor)
                cursor += len(name)
            ends = starts[1:] + [1 << 20]
            header = list(zip(names, starts, ends))
            continue
        if not line.strip():
            continue
        rows.append({name: line[start:end].strip()
                     for name, start, end in header})
    return rows


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def load_probe_h_trace(path) -> dict:
    """Read original packed probe rows, preserving the allocator's multiplier.

    Sampling needs the full per-expert Fisher vector AND packed topology. An
    expanded per-expert probe cannot establish that identity and is refused.
    """
    import pickle

    probe = pickle.loads(Path(path).read_bytes())
    stats = probe.get("stats") if isinstance(probe, dict) else None
    if not isinstance(stats, dict):
        raise RuntimeError(f"--probe {path}: no 'stats' map to read h_trace from")
    return {str(name): row for name, row in stats.items()
            if isinstance(row, dict) and row.get("_packed_experts_module")}


def sample_stack_groups(groups, probe_rows, *, profile, stack_sample: int,
                        seed: int, audit_rate: int) -> dict:
    """Draw once per profile-defined packed parameter, across all its roles.

    The same expert IDs and full-frame inclusion probabilities are persisted
    for every projection and rung. The original probe remains the allocator
    input; no per-expert expansion changes its topology or Fisher currency.
    """
    from prismaquant.tessera_campaign import (
        audit_subsample, draw_stack_sample, stack_sample_from_probe,
        _validate_stack_sample, selection_stack_samples)

    sampled = {}
    for key, members in sorted(groups.items()):
        if not str(key).startswith("s:"):
            continue
        records, drawn, audit, pi = {}, set(), set(), {}
        for name, row in sorted(probe_rows.items()):
            if "s:" + str(row.get("_packed_experts_module")) != key:
                continue
            frame = stack_sample_from_probe(
                name, row, profile, sampled_experts=range(int(row["num_experts"])),
                inclusion_prob={e: 1.0 for e in range(int(row["num_experts"]))},
                seed=seed, design="census")
            _validate_stack_sample(frame)
            draw = draw_stack_sample(
                {str(e): h for e, h in enumerate(frame.h_trace_per_expert)},
                stack_sample, seed=seed, stack=name)
            audit_ids = audit_subsample(draw["units"], rate=audit_rate,
                                       seed=seed, stack=name)
            experts = sorted(int(e) for e in draw["units"])
            # Only fields read by the constructor: JSON-portable values copied
            # exactly from the probe, rather than a second normalized weight.
            probe_row = {
                "_packed_experts_module": frame.packed_experts_module,
                "_packed_param": frame.packed_param,
                "num_experts": frame.num_experts,
                "h_trace": frame.stack_h_trace,
                "h_trace_per_expert": list(frame.h_trace_per_expert),
            }
            records[name] = {
                "probe_row": probe_row, "sampled_experts": experts,
                "inclusion_prob": dict(draw["inclusion_probability"]),
                "seed": seed, "design": draw["method"], "draw": draw,
                "audit_experts": sorted(int(e) for e in audit_ids),
            }
            for expert, names in frame.members.items():
                for member in names:
                    pi[member] = draw["inclusion_probability"][str(expert)]
                    if expert in experts:
                        drawn.add(member)
                    if str(expert) in audit_ids:
                        audit.add(member)
        if not records:
            raise RuntimeError(
                f"anchor group {key}: original packed probe rows with "
                "h_trace_per_expert are required; expanded probes cannot price stacks")
        entry = {"key": key, "members": sorted(members),
                 "sampled": sorted(drawn), "audit": sorted(audit),
                 "inclusion_probability": dict(sorted(pi.items())),
                 "stack_samples": records}
        selection_stack_samples({"groups": [entry]}, profile)
        sampled[key] = {k: v for k, v in entry.items() if k not in ("key", "members")}
    return sampled


#: The two selection schemas, spelled here so that ``plan`` stays a CPU-side
#: step: importing ``prismaquant.tessera_campaign`` for two strings would drag
#: torch into a command whose whole job is to write JSON and a manifest.  They
#: are pinned to the campaign's own constants by
#: ``test_the_planner_and_the_campaign_agree_on_the_selection_schemas``, which
#: runs where the package is importable.
UNITS_SCHEMA = "prismaquant.tessera_campaign_units.v1"
UNITS_SCHEMA_V2 = "prismaquant.tessera_campaign_units.v2"


def cmd_plan(args) -> int:
    spec = load_spec(Path(args.spec))
    workspace = Path(args.workspace)
    census = json.loads((workspace / "census.json").read_text())
    if census.get("model") != spec["model"]:
        raise RuntimeError(
            f"census was taken on {census.get('model')!r}, the spec names "
            f"{spec['model']!r}")
    groups = census["anchor_groups"]
    if not groups:
        raise RuntimeError("census reports no anchor group to price")

    stack_sample: dict[str, dict] = {}
    if args.stack_sample is not None:
        if not args.probe:
            raise RuntimeError(
                "--stack-sample needs --probe: the draw is proportional to "
                "the packed probe's per-expert h_trace")
        from prismaquant.model_profiles import detect_profile
        stack_sample = sample_stack_groups(
            groups, load_probe_h_trace(args.probe), profile=detect_profile(spec["model"]),
            stack_sample=int(args.stack_sample), seed=int(args.stack_sample_seed),
            audit_rate=int(args.audit_rate))
        priced = sum(len(entry["sampled"]) for entry in stack_sample.values())
        frame = sum(len(groups[key]) for key in stack_sample)
        print(f"[dispatch] sampled {priced} of {frame} routed expert units "
              f"across {len(stack_sample)} stack(s), "
              f"{sum(len(e['audit']) for e in stack_sample.values())} audited")

    units_dir = workspace / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(groups)
    bundles = [ordered[index:index + args.groups_per_row]
               for index in range(0, len(ordered), args.groups_per_row)]

    rows: list[dict] = []
    planned: list[dict] = []
    for index, bundle in enumerate(bundles):
        row_id = f"row-{index:04d}"
        entries = []
        for key in bundle:
            entry = {"key": key, "members": sorted(groups[key])}
            if key in stack_sample:
                entry.update(stack_sample[key])
            entries.append(entry)
        selection = {
            # A file that samples says so in its schema; one that does not
            # stays byte-identical to what every row before 2026-09-06 read.
            "schema": (UNITS_SCHEMA_V2 if stack_sample else UNITS_SCHEMA),
            "model": spec["model"],
            "layer_stride": census["layer_stride"],
            "groups": entries,
        }
        units_path = units_dir / f"{row_id}.json"
        units_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
        row_dir = workspace / "rows" / row_id
        members = [name for entry in entries
                   for name in (entry.get("sampled") or entry["members"])]
        argv = [
            "--model", spec["model"],
            "--out", str(row_dir / "cost.pkl"),
            "--cache-dir", str(row_dir / "cache"),
            "--checkpoint", str(row_dir / "cost.anchors.json"),
            "--units", str(units_path),
            "--calibration-census", str(workspace / "census.json"),
            *spec["campaign_argv"],
        ]
        if args.seed_checkpoint:
            argv += ["--seed-checkpoint", str(args.seed_checkpoint)]
            if args.seed_wire_dir:
                argv += ["--seed-wire-dir", str(args.seed_wire_dir)]
        rows.append(_row(spec, argv,
                         mem_gb=_row_memory_gb(spec, members, census),
                         timeout_s=int(args.timeout_s)))
        planned.append({"row_id": row_id, "groups": bundle, "members": sorted(members),
                        "dir": str(row_dir), "units": str(units_path)})

    # The dominant term in a row's demand today is the whole checkpoint every
    # row loads, which is what a quantum holding only its own units' weights
    # would remove; until then that term is the concurrency ceiling.
    per_box = int(args.rows_per_box)
    require_rows_fit([int(row["demand"]["mem_gb"]) for row in rows],
                     per_box, spec.get("box_memory_gb"))

    manifest = workspace / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    plan = {
        "schema": PLAN_SCHEMA,
        "model": spec["model"],
        "census": str(workspace / "census.json"),
        "manifest": str(manifest),
        "groups_per_row": int(args.groups_per_row),
        "rows_per_box": per_box,
        "row_memory_gb": {row_id: int(row["demand"]["mem_gb"])
                          for row_id, row in zip(
                              (entry["row_id"] for entry in planned), rows)},
        "seed_checkpoint": (None if not args.seed_checkpoint
                            else str(args.seed_checkpoint)),
        # The draw itself, whole: which experts stand for their stack, under
        # what inclusion probability, from which probe and which seed. It is
        # here as well as in every units file because the plan is the thing a
        # reader audits, and an estimate built on a sample is only checkable
        # against the pi it was drawn under.
        "stack_sample": {
            "size": (None if args.stack_sample is None else int(args.stack_sample)),
            "seed": int(args.stack_sample_seed),
            "audit_rate": int(args.audit_rate),
            "probe": (None if not args.probe else str(args.probe)),
            "stacks": stack_sample,
        },
        "rows": planned,
    }
    (workspace / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"[dispatch] planned {len(rows)} rows over {len(ordered)} anchor groups "
          f"-> {manifest}")
    return 0


def cmd_submit(args) -> int:
    workspace = Path(args.workspace)
    # Re-running the manifest IS the resume: a finished row is a cache hit and
    # a running row is re-attached, both by pbcampaign itself.
    return _pbcampaign(workspace / "manifest.json", wait_s=args.wait_s,
                       receipts=workspace / "receipts.json")


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def _require_equal(field: str, values: dict) -> object:
    distinct = {json.dumps(value, sort_keys=True, default=str) for value in values.values()}
    if len(distinct) != 1:
        detail = ", ".join(f"{row}={json.dumps(value, sort_keys=True, default=str)[:160]}"
                           for row, value in sorted(values.items()))
        raise MergeRefused(f"rows disagree on {field}: {detail}")
    return next(iter(values.values()))


def _hessian_identities(payload: dict) -> list[dict]:
    return [row["hessian_identity"]
            for rows in payload["costs"].values() for row in rows.values()
            if "hessian_identity" in row]


def merge_payloads(row_payloads: dict, *, census: dict, capture_sha256: str) -> dict:
    """One cost payload from N rows, refusing anything they do not share.

    The merged table is the monolith's on every field the monolith's rows would
    carry: the union of the per-unit prices, one Hessian identity, and a
    coverage block rebuilt over the **scope** rather than over any one row's
    selection.
    """
    from prismaquant.tessera_campaign import (
        SCHEMA, campaign_population_block, canonical_refusals, selection_stack_samples)
    from prismaquant.tessera_campaign import ExpertPopulation
    # The keys a merged payload must land under are the ones the campaign and
    # the allocation share.  Spelling them here as literals is how a merge
    # writes a block nothing reads: POPULATION_KEY is "population", not
    # "tessera_population", and the allocation reads only the former.
    from prismaquant.tessera_expert_projection import (
        EXPERT_WIRES_KEY, POPULATION_KEY, PROJECTION_KEY)

    for row_id, payload in row_payloads.items():
        if payload.get("schema") != SCHEMA:
            raise MergeRefused(f"{row_id}: not a {SCHEMA} payload")

    provenances = {row: payload["provenance"] for row, payload in row_payloads.items()}
    for field in SHARED_PROVENANCE:
        _require_equal(f"provenance.{field}",
                       {row: prov.get(field) for row, prov in provenances.items()})
    _require_equal("provenance.hessian.calibration_identity",
                   {row: prov["hessian"]["calibration_identity"]
                    for row, prov in provenances.items()})
    _require_equal("provenance.activation_static_scales.policy",
                   {row: prov["activation_static_scales"]["policy"]
                    for row, prov in provenances.items()})
    _require_equal("currency", {row: payload["currency"]
                                for row, payload in row_payloads.items()})
    scope = _require_equal(
        "provenance.campaign_scope",
        {row: {key: value for key, value in prov["campaign_scope"].items()}
         for row, prov in provenances.items()})

    # One Hessian identity across every priced row, on every field but the
    # capture digest the merge is about to replace.
    for row_id, payload in row_payloads.items():
        for field in SHARED_HESSIAN:
            _require_equal(
                f"hessian_identity.{field}",
                {f"{row_id}:{index}": identity.get(field)
                 for index, identity in enumerate(_hessian_identities(payload))}
                or {row_id: None})
    for field in SHARED_HESSIAN:
        _require_equal(
            f"hessian_identity.{field} across rows",
            {row: (_hessian_identities(payload)[0].get(field)
                   if _hessian_identities(payload) else None)
             for row, payload in row_payloads.items()})

    # Coverage: every group in the scope priced exactly once.
    selected: dict[str, str] = {}
    selection_entries = {}
    for row_id, prov in provenances.items():
        for entry in prov["unit_selection"]["groups"]:
            key = entry["key"]
            if key in selected:
                raise MergeRefused(
                    f"anchor group {key!r} is priced by both {selected[key]} and {row_id}")
            if key not in scope["anchor_groups"] or sorted(entry["members"]) != sorted(scope["anchor_groups"][key]):
                raise MergeRefused(f"{row_id}: selection {key!r} differs from campaign scope")
            selected[key] = row_id
            selection_entries[key] = entry
    missing = sorted(set(scope["anchor_groups"]) - set(selected))
    if missing:
        raise MergeRefused(
            f"the rows do not cover {len(missing)} anchor group(s) of the scope: "
            + ", ".join(missing[:8]))

    sampled = any("stack_samples" in entry or entry.get("sampled")
                  for entry in selection_entries.values())
    merged_selection = {
        "schema": "prismaquant.tessera_campaign_units.v2" if sampled else "prismaquant.tessera_campaign_units.v1",
        "selected": False,
        "groups": [selection_entries[key] if sampled else
                   {"key": key, "members": list(scope["anchor_groups"][key])}
                   for key in sorted(selection_entries)],
    }
    stack_samples, profile = {}, None
    if sampled:
        from prismaquant.model_profiles import detect_profile
        profile = detect_profile(next(iter(provenances.values()))["model"])
        stack_samples = selection_stack_samples(merged_selection, profile)
    costs: dict[str, dict] = {}
    loo: dict[str, dict] = {}
    surfaces: dict[str, dict] = {}
    anchor_counts: dict[str, dict] = {}
    menu_sizes: dict[str, int] = {}
    anchor_groups: dict[str, list] = {}
    non_interpolable: list[dict] = []
    expert_wires: dict[str, dict] = {}
    # Evidence, not prices: rows a shard adopted from another campaign whose
    # rungs its menu does not admit.  The union is taken here for the same
    # reason the prices are -- the reference row's block describes one slice.
    unservable: dict[str, dict] = {}
    formats: set[str] = set()
    stopped_early = False
    wall_seconds = 0.0
    rounds_run = 0
    seeds: list[dict] = []
    projection_block = None
    serving_by_unit: dict[str, dict] = {}
    serving_target = None
    for row_id in sorted(row_payloads):
        payload = row_payloads[row_id]
        prov = payload["provenance"]
        for qname, rows in payload["costs"].items():
            if qname in costs:
                raise MergeRefused(f"unit {qname} is priced by more than one row")
            costs[qname] = {
                fmt: {**row, "hessian_identity": {**row["hessian_identity"],
                                                  "capture_sha256": capture_sha256}}
                if "hessian_identity" in row else row
                for fmt, row in rows.items()
            }
        formats.update(payload["formats"])
        loo.update(payload["leave_one_anchor_out"])
        non_interpolable.extend(payload["non_interpolable"])
        surfaces.update(prov["surfaces"])
        anchor_groups.update(prov["anchor_groups"])
        anchor_counts.update(payload["anchor_counts"])
        menu_sizes.update(payload["menu_sizes"])
        expert_wires.update(payload.get(EXPERT_WIRES_KEY, {}))
        for qname, rungs in (prov.get("unservable") or {}).items():
            held = unservable.setdefault(qname, {})
            for fmt, record in rungs.items():
                if fmt in held and held[fmt] != record:
                    raise MergeRefused(
                        f"{row_id}: it carries different unservable evidence for "
                        f"{qname} {fmt} than an earlier row")
                held[fmt] = record
        stopped_early = stopped_early or bool(prov["stopped_early"])
        wall_seconds += float(prov["wall_seconds"])
        rounds_run = max(rounds_run, int(prov["rounds_run"]))
        if prov.get("seed_checkpoint"):
            seeds.append({"row": row_id, **prov["seed_checkpoint"]})
        projection = prov.get(PROJECTION_KEY)
        if projection is not None:
            # Every row carries the SCOPE's projection block, because the
            # allocation rebinds the producer's answer over every stack the
            # block names. Two different blocks would be two producer answers.
            if projection_block is None:
                projection_block = projection
            elif projection_block != projection:
                raise MergeRefused(
                    f"{row_id}: its producer expert projection differs from the "
                    "other rows'; they did not read one census projection")
        serving = prov.get("tessera_serving_scope")
        if serving:
            serving_target = serving["target"] if serving_target is None else serving_target
            if serving["target"] != serving_target:
                raise MergeRefused("rows disagree on the serving target")
            serving_by_unit.update(serving["by_unit"])

    reference = provenances[sorted(provenances)[0]]
    provenance = {key: value for key, value in reference.items()}
    provenance.update({
        "surfaces": dict(sorted(surfaces.items())),
        "anchor_groups": dict(sorted(anchor_groups.items())),
        "unservable": {name: {fmt: rungs[fmt] for fmt in sorted(rungs)}
                       for name, rungs in sorted(unservable.items())},
        "stopped_early": stopped_early,
        "wall_seconds": wall_seconds,
        "rounds_run": rounds_run,
        "unit_selection": merged_selection,
        "activation_static_scales": dict(reference["activation_static_scales"]),
        "hessian": {**reference["hessian"], "capture_sha256": capture_sha256},
        "campaign_fanout": {
            "schema": PLAN_SCHEMA,
            "rows": {row_id: sorted(
                entry["key"] for entry in provenances[row_id]["unit_selection"]["groups"])
                for row_id in sorted(provenances)},
            "seed_checkpoints": seeds,
        },
    })
    if any("no_admitted_rung" in prov for prov in provenances.values()):
        provenance["no_admitted_rung"] = sorted({name for prov in provenances.values()
                                               for name in prov.get("no_admitted_rung", [])})
    if any("unit_selection_sample" in prov for prov in provenances.values()):
        audit, probabilities = set(), {}
        for row_id, prov in provenances.items():
            sample = prov.get("unit_selection_sample", {})
            audit.update(sample.get("audit_units", []))
            for name, probability in sample.get("inclusion_probability", {}).items():
                if name in probabilities and probabilities[name] != probability:
                    raise MergeRefused(f"{row_id}: different inclusion probability for {name}")
                probabilities[name] = probability
        provenance["unit_selection_sample"] = {
            "audit_units": sorted(audit), "inclusion_probability": dict(sorted(probabilities.items()))}
    if serving_target is not None:
        provenance["tessera_serving_scope"] = {
            "target": serving_target, "by_unit": dict(sorted(serving_by_unit.items()))}

    payload = {
        **{key: value for key, value in row_payloads[sorted(row_payloads)[0]].items()
           if key not in {"costs", "formats", "leave_one_anchor_out",
                          "non_interpolable", "menu_sizes", "anchor_counts",
                          "provenance", EXPERT_WIRES_KEY}},
        "schema": SCHEMA,
        "costs": dict(sorted(costs.items())),
        "formats": sorted(formats),
        "leave_one_anchor_out": dict(sorted(loo.items())),
        "non_interpolable": canonical_refusals(non_interpolable),
        "menu_sizes": dict(sorted(menu_sizes.items())),
        "anchor_counts": dict(sorted(anchor_counts.items())),
        "provenance": provenance,
    }
    if expert_wires:
        payload[EXPERT_WIRES_KEY] = dict(sorted(expert_wires.items()))
    population = ExpertPopulation(
        members=(),
        declared={stack: {name: tuple(shape) for name, shape in units.items()}
                  for stack, units in scope["declared_stacks"].items()},
        packed_in_scope={name: tuple(shape) for name, shape
                         in scope["packed_in_scope"].items()},
        omitted_outside_layer_stride={
            name: tuple(shape) for name, shape
            in scope["packed_outside_layer_stride"].items()},
    )
    # Overwrites the reference row's block, which describes that row's slice.
    payload["provenance"][POPULATION_KEY] = campaign_population_block(
        dense_targets=scope["dense_targets"], expert_targets=scope["expert_targets"],
        dense_all=scope["dense_all"], pinned=scope["pinned"],
        population=population, layer_stride=int(reference["layer_stride"]),
        costs=payload["costs"], menus=menu_sizes,
        stack_samples=stack_samples, profile=profile)
    return payload


def merge_export_inputs(row_dirs: dict, payloads: dict, *, out_cache: Path,
                        identity: dict, policy: str, static_scales: dict,
                        census: dict):
    """Union the rows' Hessian captures into the capture a whole run writes.

    The rows priced disjoint units of one draw, so their captures hold disjoint
    ``H`` under the same ``counts`` and the same provenance.  The union is
    therefore the object a whole-scope run writes, and its digest is recomputed
    from the union rather than carried over from any row.

    Four refusals stand between "the rows agree" and "this is that object":
    every row's own capture must still seal to the digest its cost rows carry;
    every row's capture provenance must be the same dict, not merely the same
    digested triple; no unit may be captured twice with different bytes; and
    the union's ``counts`` must be the census's, over the census's roster.
    """
    import torch

    from prismaquant.tessera_campaign import write_export_inputs
    from prismaquant.tessera_export_lane import hessian_capture_sha256

    hessians: dict[str, object] = {}
    counts = None
    provenance = None
    for row_id in sorted(row_dirs):
        capture = Path(row_dirs[row_id]) / "cache" / "hessian_capture.pt"
        if not capture.is_file():
            stamped = {row["hessian_identity"].get("capture_sha256")
                       for rows in payloads[row_id]["costs"].values()
                       for row in rows.values() if "hessian_identity" in row}
            if stamped - {None}:
                raise MergeRefused(
                    f"{row_id}: its rows carry a capture digest but it wrote no "
                    "Hessian capture")
            continue
        blob = torch.load(capture, map_location="cpu", weights_only=False)
        own = hessian_capture_sha256(blob["H"], blob["provenance"])
        stamped = {row["hessian_identity"].get("capture_sha256")
                   for rows in payloads[row_id]["costs"].values()
                   for row in rows.values() if "hessian_identity" in row}
        if stamped and stamped != {own}:
            raise MergeRefused(
                f"{row_id}: its cost rows carry capture digests {sorted(stamped)} "
                f"but its capture seals to {own}")
        if counts is None:
            counts, provenance = dict(blob["counts"]), dict(blob["provenance"])
        else:
            if dict(blob["counts"]) != counts:
                raise MergeRefused(
                    f"{row_id}: Hessian capture counts differ from the other rows'; "
                    "the rows did not see one calibration census")
            if dict(blob["provenance"]) != provenance:
                raise MergeRefused(
                    f"{row_id}: Hessian capture provenance differs from the other "
                    "rows'; the rows describe two calibrations")
        for name, tensor in blob["H"].items():
            if name in hessians and not torch.equal(hessians[name], tensor):
                raise MergeRefused(
                    f"{name}: two rows captured different Hessians for one unit")
            hessians[name] = tensor
    if counts is not None and dict(counts) != dict(census["counts"]):
        raise MergeRefused(
            "the merged capture's counts are not the census's; the rows did not "
            "price the scope this census describes")
    out_cache.mkdir(parents=True, exist_ok=True)
    capture_path, scales_path, capture_sha256 = write_export_inputs(
        out_cache,
        hessians=(hessians if hessians else None),
        hessian_rows=(counts or {}),
        hessian_identity=identity,
        static_scales=static_scales,
        static_scale_policy=policy,
    )
    return capture_path, scales_path, capture_sha256


def merge_checkpoint(row_dirs: dict, out_manifest: Path) -> dict:
    """One journal from the rows', under the identity their union describes.

    The rows' identities differ only where the selection does: the ``units``
    map, the serving scope's ``by_unit`` and the producer's projected stacks.
    Everything else must already be equal, and the union of the three is what a
    whole-scope run of this code computes -- so the merged journal is one a
    later whole-scope invocation can resume, and refuses by field if it is not.
    """
    from prismaquant.cost_stage_checkpoint import (
        atomic_write_bytes, canonical_json, canonical_json_sha256, unit_path,
        MANIFEST_SCHEMA, prepare_journal, write_unit,
    )

    identities = {}
    states: dict[str, dict] = {}
    stage = "Tessera campaign"
    for row_id in sorted(row_dirs):
        manifest_path = Path(row_dirs[row_id]) / "cost.anchors.json"
        manifest = json.loads(manifest_path.read_text())
        identities[row_id] = manifest["identity"]
        parts = manifest_path.with_name(manifest_path.name + ".parts")
        listed: list[str] = []
        for entry in manifest["units"]:
            qname = entry["qname"]
            listed.append(qname)
            shard = parts / entry["file"]
            if not shard.is_file():
                raise MergeRefused(
                    f"{row_id}: its journal names {qname} and the shard "
                    f"{shard} is not there; the row's anchors would be "
                    "dropped from the merged journal")
            if shard != unit_path(parts, qname):
                raise MergeRefused(f"{row_id}: unit {qname} names a noncanonical shard")
        expected = set(manifest["identity"]["units"])
        if len(listed) != len(set(listed)) or set(listed) != expected:
            raise MergeRefused(
                f"{row_id}: manifest units differ from its checkpoint identity units")
        # Reuse the journal's reader: it validates the manifest and every
        # envelope before returning state. Rehashing unchecked payload bytes
        # here would turn corrupt or foreign shards into a trusted journal.
        try:
            _, _, completed = prepare_journal(
                parts, stage=stage, resume=True, identity=manifest["identity"],
                qnames=sorted(expected), manifest_path=manifest_path)
        except RuntimeError as exc:
            raise MergeRefused(f"{row_id}: {exc}") from exc
        for qname, state in completed.items():
            if qname in states:
                raise MergeRefused(f"unit {qname} has a journal shard in two rows")
            states[qname] = state

    merged_identity = None
    for row_id, identity in sorted(identities.items()):
        if merged_identity is None:
            merged_identity = {key: value for key, value in identity.items()}
            merged_identity["units"] = dict(identity["units"])
            continue
        for key in sorted(set(merged_identity) | set(identity)):
            if key in {"units", "serving_scope", "expert_projection", "stack_sampling_identity"}:
                continue
            if (key not in merged_identity or key not in identity
                    or merged_identity[key] != identity[key]):
                raise MergeRefused(
                    f"{row_id}: checkpoint identity differs at {key!r}")
        for name, unit in identity["units"].items():
            if name in merged_identity["units"] and merged_identity["units"][name] != unit:
                raise MergeRefused(f"{row_id}: two rows bind different inputs for {name}")
            merged_identity["units"][name] = unit
        if "stack_sampling_identity" in merged_identity or "stack_sampling_identity" in identity:
            combined = dict(merged_identity.get("stack_sampling_identity", {}))
            for name, sample in identity.get("stack_sampling_identity", {}).items():
                if name in combined and combined[name] != sample:
                    raise MergeRefused(f"{row_id}: different stack_sampling_identity for {name}")
                combined[name] = sample
            merged_identity["stack_sampling_identity"] = dict(sorted(combined.items()))
        merged_identity["serving_scope"] = _merge_scope(
            merged_identity.get("serving_scope"), identity.get("serving_scope"), row_id)
        merged_identity["expert_projection"] = _merge_projection(
            merged_identity.get("expert_projection"), identity.get("expert_projection"),
            row_id)
    merged_identity["units"] = dict(sorted(merged_identity["units"].items()))

    canonical = canonical_json(merged_identity, where="merged campaign identity")
    identity_sha256 = canonical_json_sha256(canonical, where="merged campaign identity")
    parts = out_manifest.with_name(out_manifest.name + ".parts")
    for qname, state in sorted(states.items()):
        write_unit(parts, stage=stage, qname=qname,
                   identity_sha256=identity_sha256, state=state)
    manifest = {
        "schema": MANIFEST_SCHEMA, "stage": stage,
        "identity_sha256": identity_sha256, "identity": canonical,
        "units": [{"qname": qname,
                   "file": str(unit_path(parts, qname).relative_to(parts))}
                  for qname in sorted(merged_identity["units"])],
    }
    atomic_write_bytes(out_manifest, json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False).encode("utf-8"))
    return manifest


def _merge_scope(left, right, row_id):
    if left is None or right is None:
        if left != right:
            raise MergeRefused(f"{row_id}: one row has a serving scope and another does not")
        return left
    if left["target"] != right["target"]:
        raise MergeRefused(f"{row_id}: rows disagree on the serving target")
    for name in left["by_unit"].keys() & right["by_unit"].keys():
        if left["by_unit"][name] != right["by_unit"][name]:
            raise MergeRefused(f"{row_id}: different serving context for {name}")
    return {"target": left["target"],
            "by_unit": dict(sorted({**left["by_unit"], **right["by_unit"]}.items()))}


def _merge_projection(left, right, row_id):
    if left is None:
        return right
    if right is None:
        return left
    if left["source"] != right["source"]:
        raise MergeRefused(f"{row_id}: rows projected different source checkpoints")
    for name in left["stacks"].keys() & right["stacks"].keys():
        if left["stacks"][name] != right["stacks"][name]:
            raise MergeRefused(f"{row_id}: different producer projection for stack {name}")
    stacks = {**left["stacks"], **right["stacks"]}
    return {"source": left["source"], "stacks": dict(sorted(stacks.items()))}


def _require_receipts(workspace: Path, expected: int) -> None:
    """Refuse to merge a row the fleet did not report as executed.

    A ``cost.pkl`` on disk says a process wrote a file; the fleet's row table
    says which action it was, where it ran and what it exited with.  Both, or
    neither.
    """
    path = workspace / "receipts.json"
    if not path.is_file():
        raise MergeRefused(
            f"no fleet receipts at {path}; submit the manifest before merging")
    receipts = json.loads(path.read_text())
    rows = receipts.get("rows") or []
    if len(rows) != expected:
        raise MergeRefused(
            f"{path} reports {len(rows)} rows and the plan has {expected}")
    # ``pbwait.verdict`` is the fleet's own reading of the table: 0 when every
    # row's work is done, and a memoized ``cache_hit`` counts as done there.
    # It is the gate, because a re-submitted row that was already priced
    # reports no launcher status of its own and renders ``rc`` as ``-``.
    if receipts.get("returncode") not in {0, "0"}:
        raise MergeRefused(
            f"{path} records pbcampaign exit {receipts.get('returncode')!r}; "
            "not every row is done")
    failed = [row for row in rows
              if row.get("rc") not in {"0", 0, "-", ""}]
    if failed:
        raise MergeRefused(
            "the fleet reports a non-zero exit for "
            + ", ".join(f"{row.get('key')} (rc={row.get('rc')})" for row in failed))
    where = sorted({f"{row.get('host') or '?'} ({row.get('status')})"
                    for row in rows})
    print(f"[dispatch] {len(rows)} rows: " + ", ".join(where))


def cmd_merge(args) -> int:
    workspace = Path(args.workspace)
    plan = json.loads((workspace / "plan.json").read_text())
    census = json.loads(Path(plan["census"]).read_text())
    row_dirs = {entry["row_id"]: entry["dir"] for entry in plan["rows"]}
    missing = sorted(row for row, path in row_dirs.items()
                     if not (Path(path) / "cost.pkl").is_file())
    if missing:
        raise MergeRefused(
            f"{len(missing)} planned row(s) wrote no cost.pkl: " + ", ".join(missing[:8]))
    _require_receipts(workspace, len(row_dirs))
    payloads = {}
    for row_id, path in row_dirs.items():
        with open(Path(path) / "cost.pkl", "rb") as handle:
            payloads[row_id] = pickle.load(handle)

    reference = payloads[sorted(payloads)[0]]["provenance"]
    # Under a census every row calibrated the SCOPE's static scales, so this is
    # an equality check and not a union: two rows that disagree here priced two
    # different A-side contracts for one fused module.
    static_scales = _require_equal(
        "provenance.activation_static_scales.units",
        {row: payload["provenance"]["activation_static_scales"]["units"]
         for row, payload in payloads.items()})
    out_cache = Path(args.out).parent / "cache"
    _capture, _scales, capture_sha256 = merge_export_inputs(
        row_dirs, payloads, out_cache=out_cache,
        identity=reference["hessian"]["calibration_identity"],
        policy=reference["activation_static_scales"]["policy"],
        static_scales=static_scales, census=census)
    merged = merge_payloads(payloads, census=census, capture_sha256=capture_sha256)
    merged["provenance"]["cache_dir"] = str(out_cache)
    merged["provenance"]["wire_dir"] = str(out_cache / "wire")
    merged["provenance"]["hessian"]["capture_path"] = (
        None if _capture is None else str(_capture))
    merged["provenance"]["activation_static_scales"]["path"] = (
        None if _scales is None else str(_scales))

    wire_out = out_cache / "wire"
    wire_out.mkdir(parents=True, exist_ok=True)
    linked = 0
    for path in sorted(row_dirs.values()):
        source_dir = Path(path) / "cache" / "wire"
        if not source_dir.is_dir():
            continue
        for blob in sorted(source_dir.iterdir()):
            target = wire_out / blob.name
            if target.exists():
                continue
            try:
                os.link(blob, target)
            except OSError:
                target.write_bytes(blob.read_bytes())
            linked += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as handle:
        pickle.dump(merged, handle)
    manifest = merge_checkpoint(row_dirs, out.with_suffix(".anchors.json"))

    from prismaquant.tessera_menu import assert_uniform_hessian_identity

    identity = assert_uniform_hessian_identity(merged["costs"])
    total = sum(len(rows) for rows in merged["costs"].values())
    print(f"[dispatch] merged {len(payloads)} rows -> {out}: "
          f"{len(merged['costs'])} units, {total} priced rungs, "
          f"{len(merged['formats'])} formats, {linked} wire blobs, "
          f"checkpoint {manifest['identity_sha256'][:12]}")
    print(f"[dispatch] one Hessian identity: capture_sha256="
          f"{identity.get('capture_sha256')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    census = sub.add_parser("census", help="take the scope's calibration census")
    census.add_argument("--spec", required=True)
    census.add_argument("--workspace", required=True)
    census.add_argument("--timeout-s", type=int, default=7200)
    census.add_argument("--wait-s", type=int, default=14400)
    census.add_argument("--submit", action="store_true")
    census.set_defaults(func=cmd_census)

    plan = sub.add_parser("plan", help="lay the campaign out as pbcampaign rows")
    plan.add_argument("--spec", required=True)
    plan.add_argument("--workspace", required=True)
    plan.add_argument("--groups-per-row", type=int, default=1)
    plan.add_argument("--rows-per-box", type=int, default=1,
                      help="how many of these rows one box is meant to run at "
                           "once. It does not change a row's demand -- PB "
                           "places on the demand, and shrinking it to force "
                           "co-residency would be reserving less than the row "
                           "holds. It is checked against the spec's "
                           "'box_memory_gb', when the spec declares one, and "
                           "recorded in the plan.")
    plan.add_argument("--timeout-s", type=int, default=14400)
    plan.add_argument("--stack-sample", type=int, default=None,
                      help="price each routed stack from this many experts "
                           "per role, drawn proportional to the probe's "
                           "h_trace. Unset prices every expert.")
    plan.add_argument("--stack-sample-seed", type=int, default=0,
                      help="the draw's seed; the same seed and the same probe "
                           "draw the same experts.")
    plan.add_argument("--audit-rate", type=int, default=10,
                      help="one sampled expert in this many gets a third "
                           "anchor and a leave-one-out check.")
    plan.add_argument("--probe", default=None,
                      help="a probe pickle carrying per-expert h_trace.")
    plan.add_argument("--seed-checkpoint", default=None,
                      help="a campaign checkpoint whose measured anchors every "
                           "row may adopt, subject to its own row gates")
    plan.add_argument("--seed-wire-dir", default=None)
    plan.set_defaults(func=cmd_plan)

    submit = sub.add_parser(
        "submit", help="submit the manifest; re-running it is the resume")
    submit.add_argument("--workspace", required=True)
    submit.add_argument("--wait-s", type=int, default=86400)
    submit.set_defaults(func=cmd_submit)

    merge = sub.add_parser("merge", help="one cost.pkl and journal from the rows")
    merge.add_argument("--workspace", required=True)
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=cmd_merge)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
