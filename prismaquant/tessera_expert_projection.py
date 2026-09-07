"""The producer's packed-source -> serving-unit projection, carried by PrismaQuant.

PrismaQuant prices routed experts as per-expert 2-D units or packed decisions
estimated from sampled experts. The source units
(``model.layers.N.feed_forward.experts.<i>.w1``) follow the model profile's
declared packed split (``lfm2_moe.json`` ``projection_splits``). Tessera, the
producer, executes those units as ONE stack per MoE block (``<block>.experts``),
and publishes exactly which source tensor, which expert, which role and which
geometry each executed unit is through ``experiments/tessera_producer_plan.py``
(schema ``tessera.expert_projection.v1``, ``tessera.serving_parts.source_identity``
for the checkpoint binding).

This module is the ONE place PrismaQuant reads that projection.  It does not
derive structure from tensor names outside the owning profile, and it never
slices a packed source tensor itself: a unit the producer projects as anything
but a whole unpacked per-expert source tensor is refused by name, because
executing ``first_half``/``second_half``/``transpose`` selectors here would be a
second home for ``export_tessera_serving.packed_expert_weight``.  Where the
producer cannot attest a unit (a layout it does not project, a stack it did not
plan) PrismaQuant refuses that unit by name rather than guessing a slice
(AGENTS.md: refuse where bytes are decided; one rule, one home).

What flows through here, in order:

* the campaign asks the producer for the projection of every in-scope stack
  (:func:`request_expert_projection`), binds it to the profile-declared units
  (:func:`bind_expert_projection`) and prices each executed unit under the
  producer's own ``unit_input_identity`` receipt;
* the allocator carries the projection block and the priced-wire receipts of
  the selected rungs into ``__prismaquant__`` unchanged;
* the export lane re-binds every selected routed unit to that projection
  (:func:`require_stack_uniform_assignment`, :func:`verify_expert_wire_record`)
  and writes the producer's ``tessera.cached_units.v1`` manifest
  (:func:`cached_units_manifest`) so the exporter packs the priced bytes
  unchanged (``--cached-expert-units``): priced == written.

RobTand/prismaquant#183.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The producer's projection schema (``export_tessera_serving.project_expert_plan``).
PROJECTION_SCHEMA = "tessera.expert_projection.v1"
#: The only source layout this bridge executes: one whole per-expert 2-D source
#: tensor per unit.  Pinned to ``tessera.serving.scheme.MOE_SOURCE_UNPACKED`` by
#: ``tests/test_tessera_expert_projection.py``; spelled here so the export gate
#: can refuse without importing the producer package.
SOURCE_LAYOUT_UNPACKED = "unpacked_per_expert"
#: The producer's whole-tensor selector for an unpacked unit.
WHOLE_SELECTOR = "whole"
#: The producer tool this bridge shells out to, as declared in
#: ``lane_specs/tessera.json`` ``producer_tools``.
PRODUCER_PLAN_TOOL = "experiments/tessera_producer_plan.py"
#: The keys of a producer unit record that ``tessera.cached_unit.unit_input_identity``
#: seals into the priced-wire receipt.  Pinned against the producer by test.
UNIT_IDENTITY_KEYS = ("cols", "expert", "group", "projection", "rows",
                      "source_layout", "source_slice", "source_tensor", "tensor")
#: The keys ``tessera.serving_parts.source_identity`` publishes.
SOURCE_IDENTITY_KEYS = ("auxiliary_sha256", "config_sha256", "files", "tensors")

#: Where the campaign payload and the allocation carry the projection.
PROJECTION_KEY = "tessera_expert_projection"
#: Where the campaign payload and the allocation carry the priced-wire receipts
#: of projected expert units (payload: every priced rung; allocation: the
#: selected rung only).
EXPERT_WIRES_KEY = "tessera_expert_wires"
#: Where the campaign payload says which population it priced and omitted.
POPULATION_KEY = "population"
POPULATION_SCHEMA = "prismaquant.tessera_campaign_population.v2"
LEGACY_POPULATION_SCHEMA = "prismaquant.tessera_campaign_population.v1"
#: The projection block's own envelope schema inside PrismaQuant artifacts.
CARRIED_PROJECTION_SCHEMA = "prismaquant.tessera_expert_projection.v1"


class ExpertProjectionError(RuntimeError):
    """The producer's projection is absent, malformed, or does not cover a unit."""


# ---------------------------------------------------------------------------
# Asking the producer
# ---------------------------------------------------------------------------
def producer_plan_tool(env: Mapping[str, str] | None = None) -> Path:
    """The declared producer projection tool, located through the lane spec.

    The lane spec's ``producer_tools`` roster is the one list of Tessera files
    this repository shells out to; a tool absent from it is one nobody can
    check for.  ``TESSERA_REPO`` locates the pinned checkout, as it does for
    the translator and the exporter.
    """
    from .tessera_export_lane import TesseraExportLaneError, require_producer_tools

    try:
        resolved = require_producer_tools(env=env)
    except TesseraExportLaneError as exc:
        raise ExpertProjectionError(str(exc)) from exc
    for path in resolved:
        if path.endswith("/" + PRODUCER_PLAN_TOOL):
            return Path(path)
    raise ExpertProjectionError(
        f"lane_specs/tessera.json producer_tools does not declare {PRODUCER_PLAN_TOOL}; "
        "the packed-expert bridge needs the producer's explicit projection and "
        "will not derive one from tensor names")


def stack_plan_request(stacks: Mapping[str, tuple[str, int]]) -> dict:
    """The producer's stack-plan shape: ``{stack: {grid, q256, source_layout}}``.

    ``project_expert_plan`` requires exactly these keys with an int ``q256``.
    The projection's unit records do not depend on the rung -- it only checks
    the family route -- so the campaign asks once per stack at one legal rung
    and prices every rung against the same records.
    """
    plan = {}
    for stack, (grid, q256) in sorted(stacks.items()):
        if not isinstance(stack, str) or not stack:
            raise ExpertProjectionError("stack plan request needs non-empty stack names")
        if not isinstance(grid, str) or not grid:
            raise ExpertProjectionError(f"{stack}: stack plan request needs a grid name")
        if type(q256) is not int or q256 <= 0:
            raise ExpertProjectionError(f"{stack}: stack plan request needs a positive int q256")
        plan[stack] = {"grid": grid, "q256": q256, "source_layout": SOURCE_LAYOUT_UNPACKED}
    return plan


def request_expert_projection(model_path: str | Path, stacks: Mapping[str, tuple[str, int]],
                              *, out_path: str | Path, env: Mapping[str, str] | None = None,
                              python: str | None = None) -> dict:
    """Run the producer's projection tool ONCE for every requested stack.

    ``source_identity`` hashes every checkpoint file, so this is one subprocess
    per campaign (all stacks in scope), not one per stack.  The request is
    written beside the answer so a reader can see what was asked.
    """
    if not stacks:
        raise ExpertProjectionError("no stacks to project")
    tool = producer_plan_tool(env=env)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    request = out.with_name(out.name + ".request.json")
    request.write_text(json.dumps(stack_plan_request(stacks), indent=1, sort_keys=True))
    command = [python or sys.executable, str(tool), str(model_path),
               "--stack-plan", str(request), "--out", str(out)]
    completed = subprocess.run(
        command, env=dict(os.environ if env is None else env),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-12:])
        raise ExpertProjectionError(
            f"producer projection tool failed (exit {completed.returncode}): "
            f"{' '.join(command)}\n{tail}")
    try:
        projection = json.loads(out.read_text())
    except (OSError, ValueError) as exc:
        raise ExpertProjectionError(f"producer projection unreadable at {out}: {exc}") from exc
    return projection


# ---------------------------------------------------------------------------
# Binding the projection to the profile-declared units
# ---------------------------------------------------------------------------
def unit_name_of(tensor: str) -> str:
    """``ActivationSource.unit_name``: the logical tensor name without ``.weight``."""
    if not isinstance(tensor, str) or not tensor.endswith(".weight") or len(tensor) <= 7:
        raise ExpertProjectionError(f"producer unit tensor must end with .weight: {tensor!r}")
    return tensor[:-len(".weight")]


def _require_source_identity(source: Any) -> dict:
    if not isinstance(source, Mapping) or set(source) != set(SOURCE_IDENTITY_KEYS):
        raise ExpertProjectionError(
            "producer projection source identity must carry exactly "
            f"{sorted(SOURCE_IDENTITY_KEYS)}")
    for key in ("config_sha256",):
        if not isinstance(source[key], str) or not source[key]:
            raise ExpertProjectionError(f"producer projection source.{key} must be a sha256")
    for key in ("auxiliary_sha256", "files", "tensors"):
        if not isinstance(source[key], Mapping):
            raise ExpertProjectionError(f"producer projection source.{key} must be an object")
    for tensor, file in source["tensors"].items():
        if not isinstance(tensor, str) or not isinstance(file, str) or file not in source["files"]:
            raise ExpertProjectionError(
                f"producer projection source.tensors[{tensor!r}] must name a hashed file")
    return dict(source)


def _validate_unit(stack: str, record: Any, declared_units: Mapping[str, tuple[int, int]],
                   tensors: Mapping[str, str]) -> tuple[str, dict]:
    if not isinstance(record, Mapping):
        raise ExpertProjectionError(f"{stack}: producer unit record is not an object")
    missing = set(UNIT_IDENTITY_KEYS) - set(record)
    if missing:
        raise ExpertProjectionError(
            f"{stack}: producer unit record lacks {sorted(missing)}")
    name = unit_name_of(record["tensor"])
    if record["source_layout"] != SOURCE_LAYOUT_UNPACKED:
        raise ExpertProjectionError(
            f"{name}: producer projects source layout {record['source_layout']!r}; this "
            f"bridge executes only {SOURCE_LAYOUT_UNPACKED!r} units, because slicing a "
            "packed source tensor here would be a second home for the producer's "
            "packed_expert_weight")
    if record["source_tensor"] != record["tensor"]:
        raise ExpertProjectionError(
            f"{name}: producer source tensor {record['source_tensor']!r} is not the unit "
            "tensor; an unpacked unit is its own whole source tensor")
    expert = record["expert"]
    if type(expert) is not int or expert < 0:
        raise ExpertProjectionError(f"{name}: producer expert id must be a non-negative int")
    expected_slice = {"expert": expert, "selector": WHOLE_SELECTOR, "transpose": False}
    if record["source_slice"] != expected_slice:
        raise ExpertProjectionError(
            f"{name}: producer source slice {record['source_slice']!r} is not the whole "
            f"unpacked tensor {expected_slice!r}")
    for key in ("projection", "group"):
        if not isinstance(record[key], str) or not record[key]:
            raise ExpertProjectionError(f"{name}: producer unit {key} must be a non-empty string")
    if name not in declared_units:
        raise ExpertProjectionError(
            f"{name}: producer projects a unit the profile does not declare in stack {stack}")
    rows, cols = record["rows"], record["cols"]
    if type(rows) is not int or type(cols) is not int or rows <= 0 or cols <= 0:
        raise ExpertProjectionError(f"{name}: producer rows/cols must be positive ints")
    if (rows, cols) != tuple(declared_units[name]):
        raise ExpertProjectionError(
            f"{name}: producer geometry [{rows}, {cols}] disagrees with the declared source "
            f"unit {list(declared_units[name])}")
    if record["source_tensor"] not in tensors:
        raise ExpertProjectionError(
            f"{name}: producer source tensor is not in the hashed checkpoint roster")
    return name, {key: record[key] for key in UNIT_IDENTITY_KEYS}


def bind_expert_projection(projection: Any, *,
                           declared: Mapping[str, Mapping[str, tuple[int, int]]],
                           allow_unrequested_stacks: bool = False,
                           ) -> dict[str, dict[str, dict]]:
    """Bind the producer's projection to PrismaQuant's declared units, exactly.

    ``declared`` is ``{stack: {unit_qname: (rows, cols)}}`` from the profile's
    packed split -- the units PrismaQuant prices or selected.  Every declared
    stack must be projected, and each projected stack must cover exactly its
    declared units with the same geometry: a stack executes whole, so a
    projection covering more or fewer experts than the profile declares is a
    projection of a different tensor.  Returns ``{stack: {unit_qname: unit
    record}}`` with the record trimmed to the keys the producer seals.
    """
    if not isinstance(projection, Mapping):
        raise ExpertProjectionError("producer projection is not an object")
    if projection.get("schema") != PROJECTION_SCHEMA:
        raise ExpertProjectionError(
            f"producer projection schema {projection.get('schema')!r} is not "
            f"{PROJECTION_SCHEMA!r}")
    if not {"schema", "stacks", "source"} <= set(projection):
        raise ExpertProjectionError("producer projection must carry schema, stacks and source")
    source = _require_source_identity(projection["source"])
    stacks = projection["stacks"]
    if not isinstance(stacks, Mapping):
        raise ExpertProjectionError("producer projection stacks must be an object")
    if not declared:
        raise ExpertProjectionError("no declared stacks to bind")
    missing = sorted(set(declared) - set(stacks))
    if missing:
        raise ExpertProjectionError(
            f"producer projection does not plan stacks {missing}; their units are refused")
    extra = sorted(set(stacks) - set(declared))
    if extra and not allow_unrequested_stacks:
        raise ExpertProjectionError(
            f"producer projection plans stacks that were not requested: {extra}")
    bound: dict[str, dict[str, dict]] = {}
    for stack in sorted(declared):
        entry = stacks[stack]
        if not isinstance(entry, Mapping) or not isinstance(entry.get("units"), list):
            raise ExpertProjectionError(f"{stack}: producer stack entry lacks a units list")
        if entry.get("source_layout") != SOURCE_LAYOUT_UNPACKED:
            raise ExpertProjectionError(
                f"{stack}: producer stack source layout {entry.get('source_layout')!r} is "
                f"not {SOURCE_LAYOUT_UNPACKED!r}; this bridge does not slice packed sources")
        units: dict[str, dict] = {}
        experts: set[int] = set()
        for record in entry["units"]:
            name, trimmed = _validate_unit(stack, record, declared[stack], source["tensors"])
            if name in units:
                raise ExpertProjectionError(f"{name}: producer projects the unit twice")
            units[name] = trimmed
            experts.add(int(trimmed["expert"]))
        undeclared = sorted(set(declared[stack]) - set(units))
        if undeclared:
            raise ExpertProjectionError(
                f"{stack}: producer projection does not cover declared units {undeclared}")
        # The producer states the stack's expert COUNT (``plan_expert_stack``
        # has already refused a gap or an undeclared index against
        # config.json); its units must then be exactly ``range(count)``.
        planned = entry.get("experts")
        if type(planned) is not int or sorted(experts) != list(range(planned)):
            raise ExpertProjectionError(
                f"{stack}: producer stack experts {planned!r} disagree with its units' "
                f"experts {sorted(experts)}")
        bound[stack] = units
    return bound


def carried_projection(projection: Mapping[str, Any], bound: Mapping[str, Mapping[str, dict]],
                       *, request: Mapping[str, Any], tool: str) -> dict:
    """The block the campaign payload and the allocation carry.

    The producer's answer is kept verbatim under ``producer`` (it is the
    producer's statement, not PrismaQuant's), beside the exact binding that
    was priced and the request that produced it.
    """
    return {
        "schema": CARRIED_PROJECTION_SCHEMA,
        "tool": str(tool),
        "request": {stack: dict(entry) for stack, entry in sorted(request.items())},
        "producer": json.loads(json.dumps(projection, sort_keys=True)),
        "stacks": {stack: {name: dict(unit) for name, unit in sorted(units.items())}
                   for stack, units in sorted(bound.items())},
    }


def carried_units(carried: Any) -> tuple[dict, dict[str, dict], dict[str, str]]:
    """Read a carried projection block back: ``(source, units, stack_of)``.

    Every unit is re-validated against the producer's own answer inside the
    block, so a hand-edited allocation cannot carry a unit the producer never
    projected.
    """
    if not isinstance(carried, Mapping) or carried.get("schema") != CARRIED_PROJECTION_SCHEMA:
        raise ExpertProjectionError(
            "allocation carries no producer expert projection "
            f"({PROJECTION_KEY}: {CARRIED_PROJECTION_SCHEMA})")
    stacks = carried.get("stacks")
    if not isinstance(stacks, Mapping) or not stacks:
        raise ExpertProjectionError("carried expert projection names no stacks")
    declared = {}
    for stack, units in stacks.items():
        if not isinstance(units, Mapping):
            raise ExpertProjectionError(f"{stack}: carried stack is not an object")
        declared[stack] = {}
        for name, unit in units.items():
            if not isinstance(unit, Mapping) or set(unit) != set(UNIT_IDENTITY_KEYS):
                raise ExpertProjectionError(f"{name}: carried unit record is not sealed")
            declared[stack][name] = (unit["rows"], unit["cols"])
    bound = bind_expert_projection(carried.get("producer"), declared=declared,
                                   allow_unrequested_stacks=True)
    for stack, units in bound.items():
        for name, unit in units.items():
            if dict(stacks[stack][name]) != unit:
                raise ExpertProjectionError(
                    f"{name}: carried unit record disagrees with the producer's projection")
    source = _require_source_identity(carried["producer"]["source"])
    units_flat = {name: unit for units in bound.values() for name, unit in units.items()}
    stack_of = {name: stack for stack, units in bound.items() for name in units}
    return source, units_flat, stack_of


# ---------------------------------------------------------------------------
# The source bytes the producer will read
# ---------------------------------------------------------------------------
def source_unit_weight(model_path: str | Path, source: Mapping[str, Any], unit: Mapping[str, Any]):
    """Read the unit's whole source tensor from the shard the producer hashed.

    The exporter re-reads exactly this tensor (``packed_expert_weight`` on an
    unpacked unit) and re-derives the cached identity from its bytes, so the
    campaign must price these bytes and nothing else.
    """
    from safetensors import safe_open

    tensor = unit["source_tensor"]
    try:
        file = source["tensors"][tensor]
    except KeyError:
        raise ExpertProjectionError(f"{tensor}: not in the producer's hashed tensor roster")
    path = Path(model_path) / file
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if tensor not in handle.keys():
            raise ExpertProjectionError(f"{tensor}: absent from {path}")
        weight = handle.get_tensor(tensor)
    if list(weight.shape) != [unit["rows"], unit["cols"]]:
        raise ExpertProjectionError(
            f"{tensor}: source shape {list(weight.shape)} disagrees with the projection "
            f"[{unit['rows']}, {unit['cols']}]")
    return weight.contiguous()


# ---------------------------------------------------------------------------
# The export side: selected units, stack-uniform rungs, priced-wire receipts
# ---------------------------------------------------------------------------
def require_stack_uniform_assignment(selected: Mapping[str, str], stack_of: Mapping[str, str],
                                     units: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """One rung per executed stack, or refuse by name.

    The producer's stack plan carries ONE ``(grid, q256)`` per stack, and the
    profile's ``format_groups`` make the allocator broadcast one format over
    the stack's members; a stack whose selected members disagree, or that is
    only partly selected, is one the producer cannot execute as planned.
    Returns ``{stack: format}`` for every stack with a selected member.
    """
    by_stack: dict[str, dict[str, str]] = {}
    for name, fmt in selected.items():
        stack = stack_of.get(name)
        if stack is None:
            raise ExpertProjectionError(
                f"{name}: selected routed expert unit is not in the carried producer projection")
        by_stack.setdefault(stack, {})[name] = fmt
    formats: dict[str, str] = {}
    for stack, members in sorted(by_stack.items()):
        planned = sorted(n for n, s in stack_of.items() if s == stack)
        unselected = sorted(set(planned) - set(members))
        if unselected:
            raise ExpertProjectionError(
                f"{stack}: the producer executes the stack whole, but "
                f"{len(unselected)} of its {len(planned)} projected units are not "
                f"selected for Tessera (first: {unselected[0]})")
        distinct = sorted(set(members.values()))
        if len(distinct) != 1:
            raise ExpertProjectionError(
                f"{stack}: selected rungs differ across the stack {distinct}; the producer "
                "plans one rung per stack (no role split of a projected stack)")
        formats[stack] = distinct[0]
    return formats


def check_expert_wire_receipt(record: Any, *, name: str, unit: Mapping[str, Any],
                              q256: int, grid: str) -> dict:
    """Check a priced-wire receipt against the unit and rung it claims, without bytes.

    The allocator applies this to the receipt of every rung it selects, so an
    allocation never carries a receipt for another unit, another rung, another
    grid or another projection; the export lane adds the byte check
    (:func:`verify_expert_wire_record`) where the bytes are about to be handed
    over.  Returns the receipt's four producer fields and nothing else.
    """
    if not isinstance(record, Mapping) or set(record) != {"file", "blob_sha256",
                                                          "blob_bytes", "identity"}:
        raise ExpertProjectionError(f"{name}: priced-wire receipt is not a producer unit record")
    identity = record["identity"]
    if not isinstance(identity, Mapping):
        raise ExpertProjectionError(f"{name}: priced-wire receipt has no identity")
    if identity.get("unit") != name:
        raise ExpertProjectionError(
            f"{name}: priced-wire receipt is for unit {identity.get('unit')!r}")
    if identity.get("projection") != {key: unit[key] for key in UNIT_IDENTITY_KEYS}:
        raise ExpertProjectionError(
            f"{name}: priced-wire receipt was sealed under a different producer projection")
    recipe = identity.get("recipe")
    if not isinstance(recipe, Mapping) or recipe.get("q256") != q256 or recipe.get("grid") != grid:
        raise ExpertProjectionError(
            f"{name}: priced-wire receipt recipe {recipe!r} is not the selected rung "
            f"(grid={grid!r}, q256={q256})")
    file = record["file"]
    if not isinstance(file, str) or Path(file).name != file or file in {".", ".."}:
        raise ExpertProjectionError(f"{name}: priced-wire receipt file is not a local leaf")
    return {key: record[key] for key in ("file", "blob_sha256", "blob_bytes", "identity")}


def verify_expert_wire_record(record: Any, *, name: str, unit: Mapping[str, Any],
                              q256: int, grid: str, wire_dir: Path) -> dict:
    """Check a carried priced-wire receipt against the unit it claims to price.

    The producer re-verifies the receipt against the identity it recomputes
    from the source bytes and the export's Hessian; this check refuses the
    cheaper contradictions first, by name: a receipt for another unit, another
    rung, another grid, another projection, or a blob that is not in the
    campaign's wire directory with the recorded bytes.
    """
    record = check_expert_wire_receipt(record, name=name, unit=unit, q256=q256, grid=grid)
    path = wire_dir / record["file"]
    if path.is_symlink() or not path.is_file() or path.resolve().parent != wire_dir.resolve():
        raise ExpertProjectionError(f"{name}: priced wire {path} is not in the wire directory")
    blob = path.read_bytes()
    if len(blob) != record["blob_bytes"] or hashlib.sha256(blob).hexdigest() != record["blob_sha256"]:
        raise ExpertProjectionError(f"{name}: priced wire {path} does not match its receipt")
    return {key: record[key] for key in ("file", "blob_sha256", "blob_bytes", "identity")}


def cached_units_manifest(source: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]],
                          *, schema: str) -> dict:
    """The producer's ``tessera.cached_units.v1`` bundle for ``--cached-expert-units``.

    ``schema`` is the producer's own constant (``tessera.cached_unit.CACHE_SCHEMA``),
    passed in by the caller that imported it; this module does not restate it.
    """
    if not records:
        raise ExpertProjectionError("no priced expert wires to bundle")
    files = [record["file"] for record in records.values()]
    if len(set(files)) != len(files):
        raise ExpertProjectionError("priced expert wires share a filename")
    return {"schema": schema, "source": dict(source),
            "units": {name: dict(record) for name, record in sorted(records.items())}}


# ---------------------------------------------------------------------------
# The allocation side: what the layer config carries from the cost table
# ---------------------------------------------------------------------------
#: Layer-config metadata keys the allocator adds beside the three carried blocks.
STACK_FORMATS_KEY = "tessera_expert_stack_formats"
WIRE_DIR_KEY = "tessera_expert_wire_dir"


def expand_stack_decision_assignment(assignment: Mapping[str, Any], population: Any,
                                     *, units: Mapping[str, Any], stack_of: Mapping[str, str],
                                     costs: Mapping[str, Any] | None = None) -> tuple[dict, dict]:
    """Resolve explicit packed decisions to producer source units in memory.

    The population's complete member map owns the expansion. Return the source
    assignment and each expanded member's packed owner, without mutating either
    the serialized assignment or its metadata. Allocation and export share the
    ownership, coverage and contradictory-assignment checks here.
    """
    projected_assignment = dict(assignment)
    decisions = population.get("stack_decisions", {}) if isinstance(population, Mapping) else {}
    if not isinstance(decisions, Mapping):
        raise ExpertProjectionError("population stack_decisions must be an object")
    member_owner = {}
    for packed, decision in sorted(decisions.items()):
        packed_parameters = population.get("enumerated", {}).get("packed_parameters", {})
        if (packed not in packed_parameters or
                packed not in population.get("enumerated", {}).get("routed_experts", ()) or
                not isinstance(decision, Mapping)):
            raise ExpertProjectionError(f"{packed}: stack decision is not an enumerated packed parameter")
        members = decision.get("members")
        sampled = decision.get("sampled_members")
        if (not isinstance(members, list) or not members or
                any(not isinstance(n, str) or not n for n in members) or
                len(set(members)) != len(members) or
                not isinstance(sampled, list) or not sampled or
                any(not isinstance(n, str) for n in sampled) or
                len(set(sampled)) != len(sampled) or not set(sampled) <= set(members)):
            raise ExpertProjectionError(f"{packed}: stack decision has invalid source members")
        if packed not in assignment:
            raise ExpertProjectionError(f"{packed}: packed stack decision is not in the assignment")
        for name in members:
            if name in member_owner:
                raise ExpertProjectionError(f"{name}: source member belongs to multiple stack decisions")
            member_owner[name] = packed
            if name not in units or stack_of[name] != decision.get("stack"):
                raise ExpertProjectionError(f"{packed}: source member {name} is outside its producer stack")
            if (costs or {}).get(name):
                raise ExpertProjectionError(f"{name}: both source member and packed decision have cost rows")
            fmt = assignment[packed]
            if name in assignment and assignment[name] != fmt:
                raise ExpertProjectionError(f"{name}: source assignment disagrees with packed decision {packed}")
            projected_assignment[name] = fmt
        projected_assignment.pop(packed, None)
    missing = sorted(set(units) - set(projected_assignment))
    if missing:
        raise ExpertProjectionError(
            f"{len(missing)} of {len(units)} projected expert units are not in the "
            f"assignment (first: {missing[0]}); the producer executes every stack whole")
    return projected_assignment, member_owner


def allocation_expert_projection_block(payload: Mapping[str, Any],
                                       assignment: Mapping[str, Any]) -> dict:
    """Carry population/projection only after every selected receipt is present."""
    return _allocation_expert_projection_block(payload, assignment, require_wires=True)


def selection_expert_projection_block(payload: Mapping[str, Any],
                                      assignment: Mapping[str, Any]) -> dict:
    """Validate selection structure without claiming selected bytes exist.

    This is only for a non-exportable materialization request. The public
    allocation gate always requires every selected receipt.
    """
    return _allocation_expert_projection_block(payload, assignment, require_wires=False)


def _allocation_expert_projection_block(payload: Mapping[str, Any],
                                        assignment: Mapping[str, Any], *,
                                        require_wires: bool) -> dict:
    """What an allocation carries about the expert population it selected from.

    A stock table (no population statement, no projection, no wires) adds no
    keys.  A campaign table's ``population`` block travels verbatim, so the
    allocation says which units were priced and which were omitted without a
    reader inferring it from row keys.  When the table also carries the
    producer's projection, every projected unit must be placed by the
    assignment or an explicit population ``stack_decisions`` member map. A
    packed decision expands only for these receipt checks; members do not
    acquire separate prices. Each executed stack must be assigned one format (the producer
    plans one rung per stack), and every Tessera rung selected for a projected
    unit must have a receipt sealed under that unit's projection and that rung
    -- refused by name otherwise.  The receipts of exactly the selected rungs
    travel with the allocation (``tessera_expert_wires``), with the campaign's
    wire directory, so the export lane hands the exporter the priced bytes and
    nothing else.  A stack kept whole at a non-Tessera format needs no receipt;
    the block records the format so the export lane sees the same decision.
    """
    from .tessera_formats import parse_tessera_format_name

    provenance = payload.get("provenance") if isinstance(payload, Mapping) else None
    if not isinstance(provenance, Mapping):
        return {}
    population = provenance.get(POPULATION_KEY)
    carried = provenance.get(PROJECTION_KEY)
    wires = payload.get(EXPERT_WIRES_KEY)
    if population is None and carried is None and wires is None:
        return {}
    block: dict[str, Any] = {}
    if population is not None:
        if not isinstance(population, Mapping) or population.get("schema") not in {
                POPULATION_SCHEMA, LEGACY_POPULATION_SCHEMA}:
            raise ExpertProjectionError(
                f"cost table population block is not {POPULATION_SCHEMA}; the allocation "
                "cannot say which population was priced")
        block[POPULATION_KEY] = json.loads(json.dumps(population, sort_keys=True))
        if population.get("schema") == POPULATION_SCHEMA:
            unpriced = population.get("unpriced")
            if not isinstance(unpriced, Mapping) or set(unpriced) != {"dense", "routed_experts"}:
                raise ExpertProjectionError("cost table population needs explicit unpriced units")
            priced = population.get("priced")
            if not isinstance(priced, Mapping):
                raise ExpertProjectionError("cost table population needs explicit priced units")
            enumerated = population.get("enumerated")
            if not isinstance(enumerated, Mapping):
                raise ExpertProjectionError("cost table population needs explicit enumerated units")
            priced_names = set()
            for kind in ("dense", "routed_experts"):
                names = priced.get(kind)
                if (not isinstance(names, list) or
                        any(not isinstance(name, str) or not name for name in names) or
                        len(set(names)) != len(names) or priced_names.intersection(names)):
                    raise ExpertProjectionError("population priced units must be unique names")
                priced_names.update(names)
            costs = payload.get("costs", {})
            if not isinstance(costs, Mapping):
                raise ExpertProjectionError("campaign cost rows must be an object")
            actual_priced = {name for name, rows in costs.items()
                             if isinstance(rows, Mapping) and rows}
            if priced_names != actual_priced:
                raise ExpertProjectionError(
                    "population priced units disagree with nonempty campaign cost rows")
            retained = []
            for kind in ("dense", "routed_experts"):
                if not isinstance(unpriced[kind], Mapping):
                    raise ExpertProjectionError(f"population unpriced.{kind} must name units")
                targets = enumerated.get(kind)
                if (not isinstance(targets, list) or
                        any(not isinstance(name, str) or not name for name in targets) or
                        len(set(targets)) != len(targets) or
                        set(targets) != set(priced[kind]) | set(unpriced[kind])):
                    raise ExpertProjectionError(
                        f"population enumerated.{kind} must equal its priced and unpriced units")
                for name, reason in sorted(unpriced[kind].items()):
                    if (not isinstance(name, str) or not name or reason not in
                            {"no_admitted_menu", "no_successful_anchor"}):
                        raise ExpertProjectionError("population has an invalid unpriced unit")
                    if name in priced_names or name in retained:
                        raise ExpertProjectionError(
                            f"population priced/unpriced units must be disjoint: {name}")
                    if assignment.get(name) != "BF16":
                        raise ExpertProjectionError(
                            f"unpriced campaign unit {name} ({reason}) must be explicitly "
                            "retained at BF16; absent or quantized assignments have no price")
                    retained.append(name)
            if retained:
                block[POPULATION_KEY]["retained_bf16"] = sorted(retained)
    if carried is None:
        if isinstance(population, Mapping) and population.get("stack_decisions"):
            raise ExpertProjectionError(
                "stack decisions have no producer projection to bind their source members")
        if wires:
            raise ExpertProjectionError(
                "cost table carries priced expert wires but no producer projection; "
                "the wires cannot be bound to any executed unit")
        return block
    _source, units, stack_of = carried_units(carried)
    projected_assignment, _owners = expand_stack_decision_assignment(
        assignment, population, units=units, stack_of=stack_of, costs=payload.get("costs", {}))
    selected = {name: str(projected_assignment[name]) for name in units}
    stack_formats = require_stack_uniform_assignment(selected, stack_of, units)
    wire_dir = provenance.get("wire_dir")
    if not isinstance(wire_dir, str) or not wire_dir:
        raise ExpertProjectionError(
            "cost table names no wire_dir for its priced expert wires")
    if wires is None and not require_wires:
        wires = {}
    if not isinstance(wires, Mapping):
        raise ExpertProjectionError(
            "cost table carries a producer projection but no priced expert wires")
    receipts: dict[str, dict] = {}
    for name, fmt in sorted(selected.items()):
        parsed = parse_tessera_format_name(fmt)
        if parsed is None:
            continue  # kept whole at a non-Tessera format: no wire to carry
        family, q256 = parsed
        per_unit = wires.get(name)
        record = per_unit.get(fmt) if isinstance(per_unit, Mapping) else None
        if record is None and not require_wires:
            continue
        if record is None:
            raise ExpertProjectionError(
                f"{name}: selected {fmt} has no priced wire receipt in the cost table; "
                "the exporter would encode bytes this allocation did not price")
        receipts[name] = check_expert_wire_receipt(
            record, name=name, unit=units[name], q256=int(q256),
            grid=family.payload_grid().name)
    block[PROJECTION_KEY] = json.loads(json.dumps(carried, sort_keys=True))
    block[EXPERT_WIRES_KEY] = receipts
    block[STACK_FORMATS_KEY] = dict(stack_formats)
    block[WIRE_DIR_KEY] = wire_dir
    return block


def declared_stacks_from_members(members: Sequence[Any]) -> dict[str, dict[str, tuple[int, int]]]:
    """``{stack: {qname: (rows, cols)}}`` from ``PackedExpertProjection`` members.

    The stack is the packed module's own name (``<block>.experts``), which is
    also the producer's stack name for the unpacked checkpoint.
    """
    declared: dict[str, dict[str, tuple[int, int]]] = {}
    for member in members:
        shape = tuple(int(dim) for dim in member.weight.shape)
        if len(shape) != 2:
            raise ExpertProjectionError(
                f"{member.qname}: declared packed projection is not 2-D: {list(shape)}")
        declared.setdefault(member.module_qname, {})[member.qname] = shape
    return declared


__all__ = [
    "CARRIED_PROJECTION_SCHEMA",
    "EXPERT_WIRES_KEY",
    "ExpertProjectionError",
    "POPULATION_KEY",
    "POPULATION_SCHEMA",
    "PRODUCER_PLAN_TOOL",
    "PROJECTION_KEY",
    "PROJECTION_SCHEMA",
    "SOURCE_IDENTITY_KEYS",
    "SOURCE_LAYOUT_UNPACKED",
    "STACK_FORMATS_KEY",
    "UNIT_IDENTITY_KEYS",
    "WIRE_DIR_KEY",
    "allocation_expert_projection_block",
    "expand_stack_decision_assignment",
    "bind_expert_projection",
    "cached_units_manifest",
    "check_expert_wire_receipt",
    "carried_projection",
    "carried_units",
    "declared_stacks_from_members",
    "producer_plan_tool",
    "request_expert_projection",
    "require_stack_uniform_assignment",
    "source_unit_weight",
    "stack_plan_request",
    "unit_name_of",
    "verify_expert_wire_record",
]
