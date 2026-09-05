"""The producer's expert projection is carried, bound and refused by name.

PrismaQuant #183: PrismaQuant prices routed experts as profile-declared
per-expert units; Tessera executes them as one stack per MoE block.  The bridge
module is the one reader of the producer's ``tessera.expert_projection.v1``
answer.  These tests pin its vocabulary to the producer's, exercise the exact
binding (schema, layout, selector, geometry, coverage), the carried block's
round trip, the stack-uniform selection rule the export lane applies, and the
priced-wire receipt check that precedes the producer's own verification.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from prismaquant import tessera_expert_projection as tep
from prismaquant.tessera_expert_projection import (
    ExpertProjectionError,
    bind_expert_projection,
    cached_units_manifest,
    carried_projection,
    carried_units,
    producer_plan_tool,
    request_expert_projection,
    require_stack_uniform_assignment,
    stack_plan_request,
    verify_expert_wire_record,
)

STACK = "model.layers.2.feed_forward.experts"
SHARD = "model.safetensors"


def _unit(expert: int, role: str, rows: int, cols: int, **overrides) -> dict:
    """A record shaped like ``export_tessera_serving.plan_expert_stack`` writes."""
    projection = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[role]
    group = "w2" if role == "w2" else "w13"
    tensor = f"{STACK}.{expert}.{role}.weight"
    record = {
        "tensor": tensor, "wire": f"{STACK}.{expert}.{role}.wire",
        "source_tensor": tensor, "source_layout": tep.SOURCE_LAYOUT_UNPACKED,
        "source_slice": {"expert": expert, "selector": "whole", "transpose": False},
        "expert": expert, "projection": projection, "group": group,
        "rows": rows, "cols": cols,
    }
    record.update(overrides)
    return record


def _projection(experts=(0, 1), *, stacks=(STACK,), n=8, k=4) -> dict:
    tensors = {}
    stack_entries = {}
    for stack in stacks:
        units = []
        for expert in experts:
            for role, (rows, cols) in (("w1", (n, k)), ("w3", (n, k)), ("w2", (k, n))):
                unit = _unit(expert, role, rows, cols)
                unit["tensor"] = unit["source_tensor"] = f"{stack}.{expert}.{role}.weight"
                units.append(unit)
                tensors[unit["tensor"]] = SHARD
        stack_entries[stack] = {
            "source_layout": tep.SOURCE_LAYOUT_UNPACKED, "grid": "E4M3", "q256": 1024,
            "experts": len(experts), "units": units,
        }
    return {
        "schema": tep.PROJECTION_SCHEMA,
        "stacks": stack_entries,
        "source": {
            "config_sha256": "c" * 64, "auxiliary_sha256": {},
            "files": {SHARD: "f" * 64}, "tensors": tensors,
        },
    }


def _declared(experts=(0, 1), *, stacks=(STACK,), n=8, k=4) -> dict:
    return {
        stack: {
            f"{stack}.{expert}.{role}": shape
            for expert in experts
            for role, shape in (("w1", (n, k)), ("w3", (n, k)), ("w2", (k, n)))
        }
        for stack in stacks
    }


# ---------------------------------------------------------------------------
# Vocabulary pinned to the producer
# ---------------------------------------------------------------------------
def test_unpacked_layout_is_the_producers_constant():
    scheme = pytest.importorskip("tessera.serving.scheme")
    if not hasattr(scheme, "MOE_SOURCE_UNPACKED"):
        pytest.skip("pinned tessera predates MOE_SOURCE_UNPACKED")
    assert tep.SOURCE_LAYOUT_UNPACKED == scheme.MOE_SOURCE_UNPACKED


def test_unit_identity_keys_are_what_the_producer_seals():
    cached_unit = pytest.importorskip("tessera.cached_unit")
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="missing") as error:
        cached_unit.unit_input_identity(torch.zeros(2, 2), {}, None, 1024)
    listed = str(error.value).split("missing", 1)[1].strip()
    assert sorted(json.loads(listed.replace("'", '"'))) == sorted(tep.UNIT_IDENTITY_KEYS)


def test_source_identity_keys_are_what_the_producer_publishes(tmp_path):
    serving_parts = pytest.importorskip("tessera.serving_parts")
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "lfm2_moe"}))
    save_file({f"{STACK}.0.w1.weight": torch.zeros(2, 2, dtype=torch.bfloat16)},
              str(tmp_path / SHARD))
    identity = serving_parts.source_identity(tmp_path)
    assert set(identity) == set(tep.SOURCE_IDENTITY_KEYS)
    assert identity["tensors"] == {f"{STACK}.0.w1.weight": SHARD}


# ---------------------------------------------------------------------------
# The declared tool
# ---------------------------------------------------------------------------
def test_lane_spec_declares_the_producer_projection_tool(tmp_path):
    from prismaquant.lane_spec import load_lane_spec

    spec = load_lane_spec("tessera")
    declared = {tool.path for tool in spec.producer_tools}
    assert tep.PRODUCER_PLAN_TOOL in declared, (
        "the packed-expert bridge shells out to the producer's projection tool; "
        "an undeclared external dependency is one nobody can check for")
    for tool in spec.producer_tools:
        (tmp_path / tool.path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / tool.path).write_text("# stub\n")
    assert producer_plan_tool(env={"TESSERA_REPO": str(tmp_path)}) == (
        tmp_path / tep.PRODUCER_PLAN_TOOL)
    (tmp_path / tep.PRODUCER_PLAN_TOOL).unlink()
    with pytest.raises(ExpertProjectionError, match="tessera_producer_plan.py"):
        producer_plan_tool(env={"TESSERA_REPO": str(tmp_path)})


def test_request_runs_the_declared_tool_once_and_keeps_the_request(tmp_path):
    from prismaquant.lane_spec import load_lane_spec

    repo = tmp_path / "repo"
    for tool in load_lane_spec("tessera").producer_tools:
        (repo / tool.path).parent.mkdir(parents=True, exist_ok=True)
        (repo / tool.path).write_text("# stub\n")
    tool = repo / tep.PRODUCER_PLAN_TOOL
    tool.write_text(
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "plan = json.load(open(args[args.index('--stack-plan') + 1]))\n"
        "out = args[args.index('--out') + 1]\n"
        "json.dump({'echo': plan, 'src': args[0]}, open(out, 'w'))\n")
    answer = request_expert_projection(
        "/model", {STACK: ("E4M3", 1024)}, out_path=tmp_path / "proj.json",
        env={**os.environ, "TESSERA_REPO": str(repo)})
    request = json.loads((tmp_path / "proj.json.request.json").read_text())
    assert request == {STACK: {"grid": "E4M3", "q256": 1024,
                               "source_layout": tep.SOURCE_LAYOUT_UNPACKED}}
    assert answer == {"echo": request, "src": "/model"}
    tool.write_text("import sys\nprint('stack refused: not a route', file=sys.stderr)\nsys.exit(3)\n")
    with pytest.raises(ExpertProjectionError, match=r"(?s)exit 3.*stack refused") as error:
        request_expert_projection(
            "/model", {STACK: ("E4M3", 1024)}, out_path=tmp_path / "proj2.json",
            env={**os.environ, "TESSERA_REPO": str(repo)})
    assert "tessera_producer_plan.py" in str(error.value)


def test_stack_plan_request_is_the_producers_exact_shape():
    assert stack_plan_request({STACK: ("E4M3", 1024)}) == {
        STACK: {"grid": "E4M3", "q256": 1024, "source_layout": tep.SOURCE_LAYOUT_UNPACKED}}
    with pytest.raises(ExpertProjectionError, match="positive int q256"):
        stack_plan_request({STACK: ("E4M3", 1024.0)})


# ---------------------------------------------------------------------------
# Exact binding
# ---------------------------------------------------------------------------
def test_bind_accepts_the_producers_whole_unpacked_units():
    bound = bind_expert_projection(_projection(), declared=_declared())
    assert set(bound) == {STACK}
    assert set(bound[STACK]) == set(_declared()[STACK])
    unit = bound[STACK][f"{STACK}.1.w2"]
    assert set(unit) == set(tep.UNIT_IDENTITY_KEYS)
    assert unit["source_slice"] == {"expert": 1, "selector": "whole", "transpose": False}
    assert (unit["rows"], unit["cols"]) == (4, 8)


@pytest.mark.parametrize("mutate, expected", [
    (lambda p: p.update(schema="tessera.expert_projection.v0"), "schema"),
    (lambda p: p["stacks"][STACK].update(source_layout="out_first_chunked"),
     "does not slice packed sources"),
    (lambda p: p["stacks"][STACK]["units"][0].update(
        source_layout="out_first_chunked",
        source_tensor=f"{STACK}.gate_up_proj",
        source_slice={"expert": 0, "selector": "first_half", "transpose": False}),
     "second home for the producer's packed_expert_weight"),
    (lambda p: p["stacks"][STACK]["units"][0].update(
        source_slice={"expert": 0, "selector": "whole", "transpose": True}),
     "not the whole unpacked tensor"),
    (lambda p: p["stacks"][STACK]["units"][0].update(rows=16),
     r"geometry \[16, 4\] disagrees with the declared source unit \[8, 4\]"),
    (lambda p: p["stacks"][STACK]["units"].pop(), "does not cover declared units"),
    (lambda p: p["stacks"][STACK]["units"][0].update(
        tensor=f"{STACK}.9.w1.weight", source_tensor=f"{STACK}.9.w1.weight"),
     "the profile does not declare"),
    (lambda p: p["stacks"][STACK].update(experts=1), "experts .* disagree"),
    (lambda p: p["stacks"][STACK].update(experts=[0, 1]), "experts .* disagree"),
    (lambda p: p["stacks"].pop(STACK), "does not plan stacks"),
    (lambda p: p["source"].pop("tensors"), "source identity must carry exactly"),
    (lambda p: p["source"]["tensors"].pop(f"{STACK}.0.w1.weight"),
     "not in the hashed checkpoint roster"),
])
def test_bind_refuses_by_name(mutate, expected):
    projection = _projection()
    mutate(projection)
    with pytest.raises(ExpertProjectionError, match=expected):
        bind_expert_projection(projection, declared=_declared())


def test_bind_refuses_unrequested_stacks_unless_the_caller_selects_a_subset():
    other = "model.layers.4.feed_forward.experts"
    projection = _projection(stacks=(STACK, other))
    with pytest.raises(ExpertProjectionError, match="not requested"):
        bind_expert_projection(projection, declared=_declared())
    bound = bind_expert_projection(projection, declared=_declared(),
                                   allow_unrequested_stacks=True)
    assert set(bound) == {STACK}


# ---------------------------------------------------------------------------
# The carried block
# ---------------------------------------------------------------------------
def test_carried_projection_round_trips_and_refuses_edits():
    projection = _projection()
    bound = bind_expert_projection(projection, declared=_declared())
    request = stack_plan_request({STACK: ("E4M3", 1024)})
    carried = carried_projection(projection, bound, request=request, tool="/repo/tool.py")
    assert carried["schema"] == tep.CARRIED_PROJECTION_SCHEMA
    assert carried["producer"] == projection
    assert carried["request"] == request
    source, units, stack_of = carried_units(json.loads(json.dumps(carried)))
    assert source == projection["source"]
    assert set(units) == set(_declared()[STACK])
    assert set(stack_of.values()) == {STACK}
    edited = json.loads(json.dumps(carried))
    edited["stacks"][STACK][f"{STACK}.0.w1"]["rows"] = 16
    with pytest.raises(ExpertProjectionError, match="disagrees"):
        carried_units(edited)
    with pytest.raises(ExpertProjectionError, match="carries no producer expert projection"):
        carried_units({"schema": "something.else"})
    with pytest.raises(ExpertProjectionError, match="carries no producer expert projection"):
        carried_units(None)


# ---------------------------------------------------------------------------
# The export side
# ---------------------------------------------------------------------------
def test_stack_uniform_assignment_refuses_role_split_partial_and_unprojected():
    _source, units, stack_of = carried_units(carried_projection(
        _projection(), bind_expert_projection(_projection(), declared=_declared()),
        request=stack_plan_request({STACK: ("E4M3", 1024)}), tool="t"))
    uniform = {name: "TESSERA_E4M3_K1_R1024" for name in units}
    assert require_stack_uniform_assignment(uniform, stack_of, units) == {
        STACK: "TESSERA_E4M3_K1_R1024"}
    split = dict(uniform)
    split[f"{STACK}.0.w2"] = "TESSERA_E4M3_K1_R768"
    with pytest.raises(ExpertProjectionError, match="rungs differ across the stack"):
        require_stack_uniform_assignment(split, stack_of, units)
    partial = dict(uniform)
    partial.pop(f"{STACK}.1.w3")
    with pytest.raises(ExpertProjectionError, match=r"executes the stack whole.*1\.w3"):
        require_stack_uniform_assignment(partial, stack_of, units)
    with pytest.raises(ExpertProjectionError, match="not in the carried producer projection"):
        require_stack_uniform_assignment(
            {**uniform, "model.layers.6.feed_forward.experts.0.w1": "TESSERA_E4M3_K1_R1024"},
            stack_of, units)


def _record(tmp_path: Path, name: str, unit: dict, *, q256=1024, grid="E4M3") -> dict:
    blob = hashlib.sha256(name.encode()).digest() * 3
    file = name.replace(".", "__") + "__TESSERA_E4M3_K1_R1024.tessera"
    (tmp_path / file).write_bytes(blob)
    return {
        "file": file, "blob_sha256": hashlib.sha256(blob).hexdigest(),
        "blob_bytes": len(blob),
        "identity": {
            "schema": "tessera.cached_unit_inputs.v1", "unit": name,
            "recipe": {"grid": grid, "q256": q256},
            "projection": {key: unit[key] for key in tep.UNIT_IDENTITY_KEYS},
        },
    }


def test_wire_receipts_are_checked_before_the_producer_sees_them(tmp_path):
    _source, units, _stack_of = carried_units(carried_projection(
        _projection(), bind_expert_projection(_projection(), declared=_declared()),
        request=stack_plan_request({STACK: ("E4M3", 1024)}), tool="t"))
    name = f"{STACK}.0.w1"
    record = _record(tmp_path, name, units[name])
    kept = verify_expert_wire_record(record, name=name, unit=units[name], q256=1024,
                                     grid="E4M3", wire_dir=tmp_path)
    assert kept == record
    with pytest.raises(ExpertProjectionError, match="not the selected rung"):
        verify_expert_wire_record(record, name=name, unit=units[name], q256=768,
                                  grid="E4M3", wire_dir=tmp_path)
    other = f"{STACK}.1.w1"
    with pytest.raises(ExpertProjectionError, match="is for unit"):
        verify_expert_wire_record(record, name=other, unit=units[other], q256=1024,
                                  grid="E4M3", wire_dir=tmp_path)
    with pytest.raises(ExpertProjectionError, match="different producer projection"):
        verify_expert_wire_record(record, name=name, unit=units[other], q256=1024,
                                  grid="E4M3", wire_dir=tmp_path)
    (tmp_path / record["file"]).write_bytes(b"not the priced bytes")
    with pytest.raises(ExpertProjectionError, match="does not match its receipt"):
        verify_expert_wire_record(record, name=name, unit=units[name], q256=1024,
                                  grid="E4M3", wire_dir=tmp_path)
    escaped = dict(record, file="../escape.tessera")
    with pytest.raises(ExpertProjectionError, match="not a local leaf"):
        verify_expert_wire_record(escaped, name=name, unit=units[name], q256=1024,
                                  grid="E4M3", wire_dir=tmp_path)


def test_cached_units_manifest_is_the_producers_bundle(tmp_path):
    source, units, _ = carried_units(carried_projection(
        _projection(), bind_expert_projection(_projection(), declared=_declared()),
        request=stack_plan_request({STACK: ("E4M3", 1024)}), tool="t"))
    records = {name: _record(tmp_path, name, unit) for name, unit in units.items()}
    manifest = cached_units_manifest(source, records, schema="tessera.cached_units.v1")
    assert set(manifest) == {"schema", "source", "units"}
    assert manifest["source"] == source
    assert set(manifest["units"]) == set(units)
    cached_unit = pytest.importorskip("tessera.cached_unit")
    assert manifest["schema"] == cached_unit.CACHE_SCHEMA
    bundle = cached_unit.CachedUnitBundle(manifest, tmp_path, set(units), source)
    assert bundle is not None
    with pytest.raises(ExpertProjectionError, match="share a filename"):
        cached_units_manifest(source, {**records, "dup": dict(records[f"{STACK}.0.w1"])},
                              schema="tessera.cached_units.v1")
