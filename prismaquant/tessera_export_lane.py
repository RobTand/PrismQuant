"""Preflight for the ``EXPORT_CONTAINER=tessera`` arm of ``run-pipeline.sh``.

The Tessera container is the third sanctioned lane (Rob, 2026-09-02:
compressed-tensors, GGUF, Tessera).  This module is the gate the arm runs
before it spends a single GPU-hour, and it is deliberately the same shape as
the R6 lane preflight two hundred lines above it in the driver: refuse up
front, against a machine-readable table, naming what has to change.

**PrismaQuant does not vendor the Tessera exporter.**  The boundary between
the two repositories crosses at exactly two objects -- the immutable pin
(``tessera_runtime/tessera_serving_runtime_pin.json``) and the contract the
plugin packages (``tessera/serving/runtime_contract.json``) -- and the lane
spec's established pattern for everything else is to NAME a Tessera-repository
tool rather than copy it (``lane_specs/tessera.json`` already names the serve
script and the route census that way).  The arm follows it: the layer_config
to plan translation is Tessera's ``experiments/plan_from_layer_config.py`` and
the encode is Tessera's ``experiments/export_tessera_serving.py``.  Copying
either here would make this repository the second place a wire recipe lives,
which is the failure mode principle 14 exists to prevent.

**Independent fail-closed gates, before encoding.**

1. :func:`require_release_pin` -- the INSTALLED Tessera must be the exact
   pinned one: the pin's commit and the SHA-256 of the packaged
   ``runtime_contract.json`` the run actually imports.  Since 2026-09-04 that
   is what immutability rests on -- Rob retired the release-tag requirement
   ("can we just pin prismaquant to latest version of tessera? then we won't
   have to keep cutting releases"), and a digest is a stronger claim than a
   tag because a tag can be moved and a stray tree cannot be hashed into
   agreement.  This refuses release exports regardless of whether development
   admission is enabled, and says so where an operator can act on it rather
   than as ``unknown export lane`` from a vocabulary check three layers up.
2. :func:`require_executes_derived_from_contract` -- principle 14.  The lane
   spec's ``served_activation_quantization.executes`` states what the serving
   runtime EXECUTES, so it is DERIVED from the ``formats[]`` table the runtime
   publishes and any disagreement refuses.  Two of our own spec files once
   disagreed about one runtime; the runtime was never ambiguous.
3. :func:`require_declared_structure` -- the checkpoint's structural class
   must be declared by the packaged contract. This is a coarse preflight,
   not permission to treat every unit of an MoE checkpoint as an expert.
4. :func:`require_producer_tools` -- every external tool the lane DECLARES it
   shells out to must exist under the env var the declaration names.  This was
   a hardcoded ``for`` loop over two paths in ``run-pipeline.sh``; it is now
   read from ``lane_specs/tessera.json``'s ``producer_tools``, which also
   carries each tool's stability and its tracking issue, so a shipping lane's
   dependency on a script with no stability promise is a value a reader and a
   gate can both see (RobTand/prismaquant#119).
5. :func:`require_producer_repo_is_pinned` -- the checkout those tools come
   from must be the SAME Tessera the pin attests: the
   ``runtime_contract.json`` it packages must hash to the pin's
   ``contract_sha256``.  Gate 1 hashes the package this process *imports*;
   without this one a run could satisfy the pin with one Tessera while a
   different checkout WROTE the wire, which is principle 8's split-brain.
6. :func:`require_serving_target` and :func:`require_assignment_scope` -- a
   scoped (v5 or later) contract
   needs an explicit runtime target. Before translation, every selected
   Tessera unit must retain the allocation's target and per-unit context,
   agree with the source header and profile topology, and resolve all regimes
   on that context. Legacy calls without scope retain their existing gates.
7. :func:`require_priced_export_inputs` -- the inputs the allocation was
   PRICED under must be the inputs the exporter is handed
   (RobTand/prismaquant#193): an H-aware allocation needs its identity-matched
   Hessian capture, a weights-only allocation refuses a stray one, and a W4A4
   selection needs a static ``input_global_scale`` for every selected unit.
   Numbered seventh because it merged beside gate 5 on separate branches;
   ``preflight`` runs it after the assignment hash and before the scope walk.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


#: ``config.json`` keys whose positive value means the checkpoint routes tokens
#: to experts.  Named rather than sniffed: every in-tree MoE architecture spells
#: the count with one of these (``num_experts`` on transformers-5 Qwen3-MoE,
#: ``n_routed_experts`` on DeepSeek-V4 and GLM, ``num_local_experts`` on
#: MiniMax/Mixtral-lineage configs), and an architecture that invents a fourth
#: spelling must be added here in the commit that declares its Tessera lane.
ROUTED_EXPERT_COUNT_KEYS = (
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
    "num_routed_experts",
)

#: The structure id this repository uses for a checkpoint with routed experts.
#: It is one of ``lane_eligibility.STRUCTURES``; admission reads whether the
#: packaged contract declares it rather than inferring support from this id.
STRUCTURE_ROUTED_MOE = "routed_moe"
STRUCTURE_DENSE = "dense"


class TesseraExportLaneError(RuntimeError):
    """The Tessera export lane refuses this run.  Always actionable."""


# ---------------------------------------------------------------------------
# Gate 1 -- the release pin
# ---------------------------------------------------------------------------
def require_release_pin() -> None:
    """Refuse unless the INSTALLED Tessera is the exact pinned runtime.

    This is the same conjunct ``tessera_render.tessera_lane_attested`` ANDs
    into producer eligibility, called here so the driver's refusal and the
    allocator's refusal are the same fact rather than two.

    The name is historical: since 2026-09-04 the pin is a commit plus the
    packaged contract's SHA-256 rather than a release tag, because Rob retired
    the tag requirement ("we won't have to keep cutting releases"). What the
    gate asserts is unchanged in kind -- an exact, immutable, reviewed runtime
    -- and the digest is what makes it enforceable here.
    """
    from . import tessera_serving_runtime_pin as pin_module

    try:
        pin_module.require_pinned_tessera_runtime()
    except pin_module.TesseraServingRuntimePinError as exc:
        raise TesseraExportLaneError(
            f"the installed Tessera is not the pinned Tessera serving "
            f"runtime: {exc}\n"
            "  Until the two agree the release export lane is declared and "
            "gated but cannot build. Explicit development admission may "
            "allocate research artifacts; it does not satisfy this gate.\n"
            "  Either install Tessera at the pinned commit, or move the pin "
            "-- ONE reviewed commit editing "
            "prismaquant/tessera_runtime/tessera_serving_runtime_pin.json's "
            "commit/version/contract_sha256 AND the three pinned constants "
            "in prismaquant/tessera_serving_runtime_pin.py, together."
        ) from None


# ---------------------------------------------------------------------------
# Gate 2 -- principle 14: `executes` is derived, never asserted
# ---------------------------------------------------------------------------
def packaged_contract_path() -> Path:
    """The contract Tessera's own plugin packages, as a real path."""
    from importlib.resources import as_file

    from . import tessera_render as tr

    with as_file(tr.tessera_serving_contract_path()) as path:
        return Path(path)


def derive_executes(
    published_formats: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    """The activation-contract globs the packaged ``formats[]`` table implies.

    One glob per published family: its ``name_pattern`` with the rate
    placeholder ``{k}`` replaced by ``*``.  Sorted, so the derivation is a
    value and not an ordering accident.
    """
    if published_formats is None:
        from .lane_eligibility import load_published_formats

        published_formats = load_published_formats(
            contract_path=packaged_contract_path())
    if not published_formats:
        raise TesseraExportLaneError(
            "the packaged Tessera runtime contract publishes no formats[] "
            "rows, so nothing can be derived from it; an absent table is "
            "UNATTESTED, not a clean bill"
        )
    derived = set()
    for family, entry in published_formats.items():
        pattern = str(entry.get("name_pattern", ""))
        if "{k}" not in pattern:
            raise TesseraExportLaneError(
                f"the packaged contract's formats[] row for {family!r} "
                f"publishes name_pattern {pattern!r}, which carries no '{{k}}' "
                "rate placeholder; the executed-contract glob cannot be "
                "derived from it"
            )
        derived.add(pattern.replace("{k}", "*"))
    return tuple(sorted(derived))


def require_executes_derived_from_contract() -> tuple[str, ...]:
    """Principle 14: refuse when the lane spec and the runtime disagree.

    The lane spec's ``served_activation_quantization.executes`` is a claim
    about another runtime, so it is either equal to what that runtime's own
    published table implies, or it is refused.  There is no third answer and
    in particular no "the prose explains the difference": a ``rationale``
    field explains, it is never the value a gate reads.
    """
    from .lane_spec import load_lane_spec

    derived = derive_executes()
    spec = load_lane_spec("tessera")
    declared = spec.served_activation_quantization
    if declared is None:
        raise TesseraExportLaneError(
            "lane_specs/tessera.json declares no "
            "served_activation_quantization, so the A-side of every Tessera "
            "rung would price to zero; that is a currency error, not a "
            "missing annotation"
        )
    if tuple(sorted(declared.executes)) != derived:
        raise TesseraExportLaneError(
            "PRINCIPLE 14: lane_specs/tessera.json declares executes="
            f"{sorted(declared.executes)} but the pinned runtime's packaged "
            f"contract publishes {list(derived)}.\n"
            "  The producer's claim about what the serving runtime executes "
            "must be DERIVED from the runtime's own machine-readable table. "
            "Re-read the table; never edit the list to silence this."
        )
    return derived


# ---------------------------------------------------------------------------
# Gate 3 -- the structures the contract declares
# ---------------------------------------------------------------------------
def model_structure(model_path: str | Path) -> str:
    """``routed_moe`` if the checkpoint routes tokens to experts, else dense.

    Read from the checkpoint's own ``config.json``, because this is a fact
    about the artifact being built and not about the architecture family: the
    ``qwen3`` profile claims both ``Qwen3ForCausalLM`` and
    ``Qwen3MoeForCausalLM``, and only one of them is a structure the Tessera
    contract declares.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise TesseraExportLaneError(
            f"{config_path} does not exist, so the checkpoint's structure "
            "cannot be read; the lane refuses rather than assuming dense"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    blocks = [config]
    for key in ("text_config", "thinker_config", "llm_config"):
        nested = config.get(key)
        if isinstance(nested, Mapping):
            blocks.append(nested)
    for block in blocks:
        for key in ROUTED_EXPERT_COUNT_KEYS:
            try:
                count = int(block.get(key) or 0)
            except (TypeError, ValueError):
                continue
            if count > 0:
                return STRUCTURE_ROUTED_MOE
    return STRUCTURE_DENSE


def require_declared_structure(model_path: str | Path) -> str:
    """Refuse a checkpoint no ADMISSIBLE cell of the packaged contract covers.

    Two refusals, and they are different facts an operator acts on differently:

    * the contract's ``structures`` vocabulary omits this class -- the runtime
      has said nothing about it, which is UNATTESTED;
    * the vocabulary names it, but every cell that covers it is refused by the
      contract's **own** ``evidence`` (``lane_eligibility.cell_evidence_admits``)
      -- the runtime measured those routes and published a defect.

    The second is why this gate reads admissible cells rather than the
    vocabulary.  Contract v17 DECLARED ``routed_moe`` while both its routed-MoE
    cells published ``evidence.smoke.status: "repetitive"`` -- a greedy smoke
    that degenerated -- so from v17 through v20 this gate refused every
    routed-MoE unit by the cells' own evidence; v20 added a BF16-source
    control sharing the symptom (``attribution: shared_with_reference``),
    which this producer read and still did not admit on (prismaquant #198).
    Reading the vocabulary alone would have let a MoE checkpoint past this
    gate on the strength of a structure name whose every cell the runtime
    itself reported as generating incorrectly.  Nothing here bans a structure
    (principle 1), and v21 (the pinned commit, Tessera #313) is that clause
    exercised: the runtime re-measured the smoke through the checkpoint's own
    chat template, both cells read ``recorded``, and the refusal lifted with
    no change to this gate -- re-pinning that contract was the review event.
    """
    from .lane_eligibility import cell_evidence_admits, load_eligibility_table

    table = load_eligibility_table(contract_path=packaged_contract_path())
    if not table.present:
        raise TesseraExportLaneError(
            "the packaged Tessera runtime contract carries no lane_eligibility "
            "table, so no structure is attested; absence is UNATTESTED"
        )
    structure = model_structure(model_path)
    if structure not in table.structures:
        raise TesseraExportLaneError(
            f"this checkpoint's structure is {structure!r} and the pinned "
            f"Tessera runtime declares structures {list(table.structures)}.\n"
            "  A checkpoint class absent from the published table is not "
            "attested. Do not replace its selected units with BF16 merely "
            "to build an artifact different from the allocation that priced it."
        )
    refusals = sorted({
        reason for cell in table.cells if cell.structure == structure
        for admits, reason in (cell_evidence_admits(cell),) if not admits
    })
    admissible = any(
        cell.structure == structure and cell_evidence_admits(cell)[0]
        for cell in table.cells
    )
    if not admissible:
        raise TesseraExportLaneError(
            f"this checkpoint's structure is {structure!r}: the pinned "
            "Tessera runtime declares it, but every cell covering it is "
            "refused by the contract's OWN published evidence.\n  "
            + "\n  ".join(refusals or ["the table carries no cell for it"])
            + "\n  This is a serving defect the runtime measured, not a ban "
            "this repository typed. Promoting the structure is a decision on "
            "the evidence -- it belongs to Rob (principle 9), not to a wider "
            "gate here."
        )
    return structure


# ---------------------------------------------------------------------------
# Gate 4 -- the external tools the arm shells out to
# ---------------------------------------------------------------------------
def require_producer_tools(
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Refuse unless every tool the lane DECLARES it shells out to is present.

    The arm calls two scripts in the Tessera repository, because a wire recipe
    with two homes is how the two halves of one format drift apart.  The cost
    of naming rather than vendoring is a dependency on a file in another
    repository -- and until 2026-09-03 that dependency was a hardcoded
    ``for`` loop in ``run-pipeline.sh`` plus a sentence in the lane spec's
    ``notes``.  Neither is something a gate can read, and neither would have
    survived a fourth lane: the loop names two paths for one lane.

    Now the roster lives in ``lane_specs/tessera.json``'s ``producer_tools``,
    where the reader who touches the lane sees it, and this gate iterates it.
    A tool declared ``unsupported_experiments`` is not refused -- the honest
    state today is that both Tessera tools live under ``experiments/`` with no
    stability promise -- but it must name a tracking issue, which
    ``LaneProducerTool.from_dict`` enforces, and it is echoed on every run so
    the debt is visible where it is being incurred (RobTand/prismaquant#119).
    """
    import os

    from .lane_spec import load_lane_spec

    env = os.environ if env is None else env
    spec = load_lane_spec("tessera")
    if not spec.producer_tools:
        raise TesseraExportLaneError(
            "lane_specs/tessera.json declares no `producer_tools`, but the "
            "arm shells out to Tessera's own plan translator and exporter. An "
            "undeclared external dependency is one nobody can check for, and "
            "one a tidy-up in the other repository deletes silently"
        )
    resolved: list[str] = []
    for tool in spec.producer_tools:
        root = str(env.get(tool.repo_env, "") or "").strip()
        if not root:
            raise TesseraExportLaneError(
                f"{tool.repo_env} is unset, so {tool.path} cannot be located. "
                "This repository NAMES Tessera's tools instead of vendoring "
                f"them; point {tool.repo_env} at the checkout of the pinned "
                "release."
            )
        path = Path(root.rstrip("/")) / tool.path
        if not path.is_file():
            raise TesseraExportLaneError(
                f"{path} does not exist. It is declared in "
                f"lane_specs/tessera.json's producer_tools as "
                f"stability={tool.stability!r}"
                + (f" ({tool.tracking_issue})" if tool.tracking_issue else "")
                + f": {tool.description}"
            )
        resolved.append(str(path))
    return tuple(resolved)


# ---------------------------------------------------------------------------
# Gate 5 -- the checkout that ENCODES must be the pinned Tessera
# ---------------------------------------------------------------------------
def require_producer_repo_is_pinned(
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Refuse unless the checkout that ENCODES is the Tessera the pin attests.

    Gate 1 hashes the ``tessera`` package this *process imports*; gate 4
    resolves the two encoder scripts through the env var each
    ``producer_tools`` entry names (``TESSERA_REPO``).  Nothing bound those
    two together, so a run could satisfy the pin with one Tessera on
    ``PYTHONPATH`` while a **different** checkout wrote the wire -- the
    rendering/execution split-brain principle 8 exists to stop.

    That hole is one this change OPENS rather than closes: while the pin
    carried PENDING sentinels the lane could not build at all, so the two
    Tesseras could never diverge in a run that produced bytes.  Making the pin
    admittable makes the second Tessera reachable, so the same predicate is
    applied to the checkout: the ``runtime_contract.json`` it packages must
    hash to ``pin.contract_sha256``.

    The repo-relative location is DERIVED from the installed package's own
    path tail rather than typed, and both a ``src/`` layout and a flat one are
    accepted, because asserting another repository's directory layout is the
    kind of hand-assertion principle 14 refuses.  Absence is never read as
    agreement: a checkout that packages no contract is refused.
    """
    import os

    from .lane_spec import load_lane_spec
    from .shipcard import file_sha256
    from .tessera_serving_runtime_pin import load_tessera_serving_runtime_pin

    env = os.environ if env is None else env
    pin = load_tessera_serving_runtime_pin()
    suffix = Path(*packaged_contract_path().parts[-3:])
    roots: list[str] = []
    for tool in load_lane_spec("tessera").producer_tools:
        root = str(env.get(tool.repo_env, "") or "").strip().rstrip("/")
        if not root or root in roots:
            continue
        roots.append(root)
        candidates = [Path(root) / "src" / suffix, Path(root) / suffix]
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            raise TesseraExportLaneError(
                f"${tool.repo_env}={root} packages no {suffix} -- so the "
                "checkout that would ENCODE the wire cannot be shown to be "
                "the Tessera the serving pin attests. This repository names "
                "Tessera's tools instead of vendoring them; point "
                f"{tool.repo_env} at the pinned commit "
                f"({pin.commit})."
            )
        digest = file_sha256(found)
        if digest != pin.contract_sha256:
            raise TesseraExportLaneError(
                f"${tool.repo_env}={root} is not the pinned Tessera: it "
                f"packages a contract hashing {digest}, and the pin names "
                f"{pin.contract_sha256}. The runtime that ATTESTS the route "
                "and the checkout that WRITES the bytes must be one object "
                "(principle 8); a producer-side import satisfying the pin "
                "while a second checkout encodes is exactly the split this "
                f"gate refuses. Check out {pin.commit}, or move the pin in "
                "ONE reviewed commit."
            )
    return tuple(roots)


# ---------------------------------------------------------------------------
# Runtime-scoped export -- the allocation and source, not a model-wide guess
# ---------------------------------------------------------------------------
def require_serving_target(target=None, *, table=None):
    """Validate explicit v5 target input without inventing a per-unit claim."""
    from .lane_eligibility import (
        SCOPED_LANE_SCHEMAS, legacy_runtime_scope_refusal,
        load_eligibility_table,
    )
    from .tessera_serving_scope import ServingTarget

    if table is None:
        table = load_eligibility_table(contract_path=packaged_contract_path())
    if target is None:
        if table.schema in SCOPED_LANE_SCHEMAS:
            raise TesseraExportLaneError(
                "an explicit Tessera serving target is required for a scoped "
                "(v5 or later) export; "
                "supply platform, runtime image, execution mode and residency")
        return None
    if table.schema not in SCOPED_LANE_SCHEMAS:
        raise TesseraExportLaneError(legacy_runtime_scope_refusal(table.schema))
    try:
        validated = ServingTarget(**target.as_dict())
    except (AttributeError, TypeError, ValueError) as exc:
        raise TesseraExportLaneError(f"invalid Tessera serving target: {exc}") from exc
    if validated.platform not in table.platforms:
        raise TesseraExportLaneError(
            f"Tessera target platform {validated.platform!r} is not published "
            f"by this contract: {list(table.platforms)}")
    return validated


def _source_unit_shapes(model_path: str | Path, profile,
                        shards: dict[str, str] | None = None) -> dict[str, list[tuple[str, tuple]]]:
    """Read source headers and the shared name projection; never load weights.

    ``shards``, when given, is filled with ``{tensor: shard basename}`` for
    every mapped tensor, so a carried producer roster can be checked against
    the shard each source tensor actually lives in.
    """
    from .footprint import _read_safetensors_header
    from .name_projection import MAPPED, NameProjection
    from .source_prefetch import _unique_safetensor_shards

    paths = _unique_safetensor_shards(model_path)
    if not paths:
        raise TesseraExportLaneError(
            f"no indexed or model.safetensors source checkpoint under {model_path}")
    projection = NameProjection(profile)
    by_unit: dict[str, list[tuple[str, tuple]]] = {}
    seen: set[str] = set()
    for path in paths:
        for name, metadata in _read_safetensors_header(str(path)).items():
            if name == "__metadata__":
                continue
            if name in seen:
                raise TesseraExportLaneError(
                    f"source checkpoint tensor {name!r} occurs in multiple shards")
            seen.add(name)
            projected = projection.checkpoint_to_live(name)
            if projected.outcome != MAPPED:
                continue
            unit = projection.recipe_unit(projected.target)
            shape = metadata.get("shape") if isinstance(metadata, Mapping) else None
            if not isinstance(shape, list) or any(
                    isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                    for dim in shape):
                # Zero-size ancillary tensors need not make an otherwise valid
                # checkpoint unexportable; a selected one has no valid shape.
                shape = ()
            by_unit.setdefault(unit, []).append((name, tuple(shape)))
            if shards is not None:
                shards[name] = Path(path).name
    return by_unit


def _carried_expert_projection(meta: Mapping[str, Any], selected_routed: Mapping[str, str],
                               shards: Mapping[str, str]) -> dict | None:
    """Re-bind the selected routed units to the projection the allocation carries.

    A carried projection is an UNLOCK, not a new requirement: an allocation
    that carries none keeps the pre-#183 lane exactly -- routed units resolve
    on their source-member shapes and a predicated cell is still refused for
    lacking the producer's executed-unit attestation -- so this returns
    ``None``.  What it will not do is read priced-wire receipts that are bound
    to nothing: an allocation carrying wires, stack formats or a wire
    directory with the projection stripped out is refused by name.

    When a projection IS carried, the producer's record is the only
    attestation of the executed unit: every selected routed unit must be a
    projected unit whose source tensor the producer hashed in the shard it
    actually lives in, each executed stack must be selected whole at one rung
    (the stamp the allocator wrote must agree), and every selected rung's
    priced bytes must sit in the campaign's wire directory under their
    receipt.  What comes back is the bundle the exporter's
    ``--cached-expert-units`` intake consumes.
    """
    from .tessera_expert_projection import (
        EXPERT_WIRES_KEY, PROJECTION_KEY, STACK_FORMATS_KEY, WIRE_DIR_KEY,
        ExpertProjectionError, carried_units, require_stack_uniform_assignment,
        verify_expert_wire_record,
    )
    from .tessera_formats import parse_tessera_format_name

    carried = meta.get(PROJECTION_KEY)
    if carried is None:
        orphaned = sorted(key for key in (EXPERT_WIRES_KEY, STACK_FORMATS_KEY, WIRE_DIR_KEY)
                          if meta.get(key) is not None)
        if orphaned:
            raise TesseraExportLaneError(
                f"the allocation carries {orphaned} but no producer expert projection "
                f"({PROJECTION_KEY}); priced expert wires that are bound to no executed "
                "unit cannot be handed to the exporter (PrismaQuant #183)")
        return None
    try:
        source, units, stack_of = carried_units(carried)
        for name in sorted(selected_routed):
            unit = units.get(name)
            if unit is None:
                raise ExpertProjectionError(
                    f"{name}: selected routed expert unit is not in the carried producer "
                    "projection; the producer did not project it")
            tensor = unit["source_tensor"]
            hashed = source["tensors"].get(tensor)
            if hashed != shards.get(tensor):
                raise ExpertProjectionError(
                    f"{name}: the producer hashed {tensor} in shard {hashed!r}, the source "
                    f"checkpoint holds it in {shards.get(tensor)!r}")
        stack_formats = require_stack_uniform_assignment(selected_routed, stack_of, units)
        stamped = meta.get(STACK_FORMATS_KEY)
        if stamped is not None and {k: v for k, v in stamped.items()
                                    if k in stack_formats} != stack_formats:
            raise ExpertProjectionError(
                f"the allocation's {STACK_FORMATS_KEY} stamp {stamped} disagrees with the "
                f"selected stack formats {stack_formats}")
        wire_dir = meta.get(WIRE_DIR_KEY)
        if not isinstance(wire_dir, str) or not wire_dir:
            raise ExpertProjectionError(
                f"the allocation names no {WIRE_DIR_KEY} for its priced expert wires")
        wires = meta.get(EXPERT_WIRES_KEY)
        if not isinstance(wires, Mapping):
            raise ExpertProjectionError(
                f"the allocation carries a producer projection but no {EXPERT_WIRES_KEY}")
        records = {}
        for name, fmt in sorted(selected_routed.items()):
            family, q256 = parse_tessera_format_name(fmt)
            record = wires.get(name)
            if record is None:
                raise ExpertProjectionError(
                    f"{name}: selected {fmt} has no priced wire receipt in the allocation")
            records[name] = verify_expert_wire_record(
                record, name=name, unit=units[name], q256=int(q256),
                grid=family.payload_grid().name, wire_dir=Path(wire_dir))
    except ExpertProjectionError as exc:
        raise TesseraExportLaneError(f"expert projection: {exc}") from exc
    return {"source": source, "units": records, "stacks": stack_formats,
            "wire_dir": wire_dir,
            "geometry": {name: (units[name]["rows"], units[name]["cols"])
                         for name in selected_routed}}


#: The bundle's name inside the campaign's wire directory.  One name: the
#: driver reads the path back from the build anchor rather than guessing it.
CACHED_EXPERT_UNITS_FILENAME = "cached_expert_units.json"


def write_cached_expert_units(projection: Mapping[str, Any]) -> Path:
    """Write the producer's cached-unit bundle beside the priced wires.

    The exporter reads the bundle's files from the manifest's own directory,
    so the manifest lives in the campaign's wire directory and nowhere else;
    the schema is the producer's constant, imported rather than restated, and
    a checkout whose producer has no such API cannot bundle (refused by name).
    """
    from .tessera_expert_projection import ExpertProjectionError, cached_units_manifest

    try:
        from tessera.cached_unit import CACHE_SCHEMA
    except ImportError as exc:
        raise TesseraExportLaneError(
            "cannot bundle the priced expert wires: this checkout's tessera has no "
            "cached_unit bundle API (tessera.cached_unit.CACHE_SCHEMA); the exporter's "
            "--cached-expert-units intake needs the release producer (PrismaQuant #192)"
        ) from exc
    try:
        manifest = cached_units_manifest(projection["source"], projection["units"],
                                         schema=CACHE_SCHEMA)
    except ExpertProjectionError as exc:
        raise TesseraExportLaneError(f"expert projection: {exc}") from exc
    destination = Path(projection["wire_dir"]) / CACHED_EXPERT_UNITS_FILENAME
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(destination)
    return destination


def require_assignment_scope(model_path: str | Path, assignment_path: str | Path,
                             *, target=None) -> dict | None:
    """Re-resolve selected Tessera units before the external translator runs.

    The existing translator still owns serialization and expert aggregation.
    This gate accepts exact two-dimensional source units, not a guessed slice
    of a packed source tensor.  For a dense unit these are source-member
    shapes, not a claim about a fused execution unit, so a predicated cell is
    refused.  A routed expert unit resolves the same way UNLESS the allocation
    carries the producer's own projection (PrismaQuant #183), which unlocks
    the stronger reading: the producer's record attests the executed unit's
    geometry, so a predicated cell resolves on it, and the priced bytes are
    checked against their receipts here, where they are about to be handed to
    the exporter.  See :func:`_carried_expert_projection` for what that
    binding refuses by name.
    """
    from .lane_eligibility import (
        SCOPED_LANE_SCHEMAS, QUALIFICATION_DEVICE_QUALIFIED, ROUTE_STATUS_BACKED,
        ROUTE_STATUS_BACKED_WITH_SERVE_FLAG, ServingContext, cell_matches_serving_context,
        load_eligibility_table, load_published_formats, resolve_unit_route,
        unit_structural_facts,
    )
    from .layer_config import load_assignment, read_layer_config_metadata
    from .model_profiles import detect_profile
    from .tessera_serving_scope import ServingTarget, unit_structure_from_profile

    path = packaged_contract_path()
    table = load_eligibility_table(contract_path=path)
    # An old context-free export remains the old export. Explicitly scoped
    # queries never borrow a legacy table's global runtime identity.
    if target is None and table.schema not in SCOPED_LANE_SCHEMAS:
        return None
    if target is None:
        require_serving_target(target, table=table)
    try:
        selected = {name: fmt for name, fmt in load_assignment(assignment_path).items()
                    if fmt.startswith("TESSERA_")}
        meta = read_layer_config_metadata(assignment_path)
        scope = meta.get("tessera_serving_scope")
        if not isinstance(scope, Mapping):
            raise TesseraExportLaneError(
                "allocation carries no tessera_serving_scope; re-allocate with an explicit target")
        if set(scope) != {"target", "by_unit"} or not isinstance(scope["target"], Mapping):
            raise TesseraExportLaneError(
                "allocation tessera_serving_scope requires target and by_unit objects")
        recorded_target = ServingTarget(**scope["target"])
        if recorded_target.as_dict() != target.as_dict():
            raise TesseraExportLaneError(
                "export target disagrees with the allocation target: "
                f"export={target.as_dict()}, allocation={recorded_target.as_dict()}")
        target = require_serving_target(target, table=table)
        by_unit = scope["by_unit"]
        if not isinstance(by_unit, Mapping):
            raise TesseraExportLaneError("allocation scope.by_unit must be an object")
        profile = detect_profile(str(model_path))
        shards: dict[str, str] = {}
        shapes = _source_unit_shapes(model_path, profile, shards)
        structures = {name: unit_structure_from_profile(name, profile) for name in selected}
        # The producer's projection first: it is the structural refusal, it is
        # cheaper than a route resolution, and it is what attests the executed
        # geometry the rest of this loop resolves a routed unit on.
        projection = _carried_expert_projection(
            meta, {name: fmt for name, fmt in selected.items()
                   if structures[name] == STRUCTURE_ROUTED_MOE}, shards)
        attested = projection["geometry"] if projection is not None else {}
        formats = load_published_formats(contract_path=path)
        routes = {}
        for name, fmt in sorted(selected.items()):
            geometry = attested.get(name)
            if geometry is None:
                matches = shapes.get(name, ())
                if len(matches) != 1 or len(matches[0][1]) != 2:
                    raise TesseraExportLaneError(
                        f"{name}: scoped export needs one exact 2-D source checkpoint shape; "
                        f"found {list(matches)}. Packed or aggregate source units require "
                        "the producer's explicit projection, not a guessed slice.")
                geometry = matches[0][1]
            structure = structures[name]
            expected = target.context(structure)
            context_payload = by_unit.get(name)
            if not isinstance(context_payload, Mapping):
                raise TesseraExportLaneError(f"{name}: allocation is missing per-unit serving context")
            recorded = ServingContext(**context_payload)
            if recorded != expected:
                raise TesseraExportLaneError(
                    f"{name}: allocation context disagrees with the export target or source "
                    f"structure: recorded={recorded.as_dict()}, expected={expected.as_dict()}")
            rows, columns = geometry
            facts = unit_structural_facts(
                name, fmt, is_routed_moe=structure == STRUCTURE_ROUTED_MOE,
                # Tessera's per-Linear wire has no split codebooks. Expert
                # aggregation remains the producer's job, not a local guess.
                role_split=False, in_features=columns, out_features=rows,
                published_formats=formats)
            if name not in attested:
                predicated = [cell.id for cell in table.cells
                              if cell.family == facts.payload_family and cell.covers_rung(facts)
                              and cell_matches_serving_context(cell, expected) and cell.predicates]
                if predicated:
                    raise TesseraExportLaneError(
                        f"{name}: selected route is unattested at this boundary: cells "
                        f"{predicated} carry predicates requiring the producer's "
                        "executed-unit projection; source-member dimensions do not attest "
                        "a fused execution shape")
            route = resolve_unit_route(facts, table, **target.as_dict())
            if (route.route_status not in (ROUTE_STATUS_BACKED, ROUTE_STATUS_BACKED_WITH_SERVE_FLAG)
                    or any(row.qualification != QUALIFICATION_DEVICE_QUALIFIED for row in route.regimes)):
                raise TesseraExportLaneError(
                    f"{name}: selected Tessera route is {route.route_status}: "
                    f"{route.unattested_reason or 'every regime must be device_qualified and native'}")
            routes[name] = route.as_dict()
        report = {"target": target.as_dict(), "by_unit": routes,
                  "contract": table.provenance()}
        if projection is not None:
            report["expert_projection"] = projection
        return report
    except TesseraExportLaneError:
        raise
    except (OSError, ValueError, TypeError, KeyError, struct.error) as exc:
        raise TesseraExportLaneError(f"cannot attest selected Tessera export scope: {exc}") from exc


# ---------------------------------------------------------------------------
# Gate 7 -- the inputs the allocation was PRICED under
# ---------------------------------------------------------------------------
#: The identity fields an H-aware allocation must be bound against.  Spelled
#: here rather than imported from ``tessera_hessian`` so the gate can refuse
#: on a machine that can read the metadata without loading the ``tessera``
#: package; ``tessera_hessian.HESSIAN_IDENTITY_FIELDS`` derives the same
#: triple from ``tessera.export.HESSIAN_IDENTITY`` and
#: ``test_the_priced_input_triple_matches_tesseras_roster`` pins the two
#: together.
PRICED_HESSIAN_IDENTITY_FIELDS = ("text_sha256", "fit_tokens", "fit_ids_sha256")

#: The capture-context fields Tessera's seal covers beside the triple
#: (``tessera.export.CAPTURE_CONTEXT``): two captures of one token prefix
#: differ only here when the sequence layout differs (tessera#214).
CAPTURE_CONTEXT_FIELDS = ("model", "seqlen", "source")

#: The schema string sealed into the digest, Tessera's own spelling.
HESSIAN_CAPTURE_SHA256_SCHEMA = "tessera.hessian_capture.v1"

#: The block the allocator stamps beside ``tessera_hessian`` when a Tessera
#: unit is selected: ``{"schema": ..., "units": {unit: input_global_scale}}``,
#: the static A-side scale VALUE each selected unit's cost row was priced
#: under (RobTand/prismaquant#204).
PRICED_STATIC_SCALES_SCHEMA = "prismaquant.tessera_activation_static_scales.v1"


def hessian_capture_sha256(hessians: "Mapping[str, Any]",
                           provenance: "Mapping[str, Any]") -> str:
    """The content digest that binds an allocation to one Hessian capture.

    The identity triple names the token draw, not the Hessian: two captures of
    one draw with different ``H`` (a different sequence layout, a different
    model revision, a corrupted or rewritten payload) carry the same triple
    and encode different bytes at the same format name.  This digest tells
    them apart.  It is **Tessera's own rule**, ``tessera.export.
    ActivationSource.capture_sha256`` at release e78959e, reimplemented here
    with torch alone because the pinned development Tessera (1221d2a) predates
    it and the gate must refuse on a machine that can read the payload
    without importing the ``tessera`` package:

    ``sha256(json.dumps({"schema": "tessera.hessian_capture.v1", "identity":
    {triple + capture context, via .get}}, sort_keys=True, default=str))``,
    then for every unit name in sorted order ``b"\\0" + name + b"\\0"`` and the
    hex of ``sha256(json.dumps({"dtype", "shape"}, sort_keys=True) + b"\\0" +
    contiguous bytes)`` of its ``H`` (``cached_unit.tensor_identity``,
    ``sha256.dtype_shape_contiguous.v1``).

    Where the running Tessera publishes ``capture_sha256`` the gate computes
    both and refuses on disagreement, so a drift in either rule is a refusal
    here rather than a silent divergence
    (``test_the_digest_rule_is_tesseras_capture_seal``).
    """
    import hashlib

    import torch

    identity = {field: provenance.get(field)
                for field in PRICED_HESSIAN_IDENTITY_FIELDS}
    identity.update({field: provenance.get(field)
                     for field in CAPTURE_CONTEXT_FIELDS})
    digest = hashlib.sha256()
    digest.update(json.dumps({"schema": HESSIAN_CAPTURE_SHA256_SCHEMA,
                              "identity": identity},
                             sort_keys=True, default=str).encode())
    for name in sorted(hessians):
        value = hessians[name].detach().cpu().contiguous()
        unit = hashlib.sha256()
        unit.update(json.dumps({"dtype": str(value.dtype),
                                "shape": list(value.shape)},
                               sort_keys=True).encode())
        unit.update(b"\0")
        unit.update(value.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0" + name.encode() + b"\0")
        digest.update(unit.hexdigest().encode())
    return digest.hexdigest()


def _tessera_capture_seal(hessians, provenance) -> "str | None":
    """Tessera's own ``capture_sha256`` of the same payload, or None where the
    running Tessera predates the seal (the pinned 1221d2a does)."""
    try:
        from tessera.export import ActivationSource
    except ImportError:
        return None
    if not hasattr(ActivationSource, "capture_sha256"):
        return None
    try:
        source = ActivationSource(hessians=hessians, provenance=dict(provenance))
        return str(source.capture_sha256())
    except Exception as exc:  # tessera's GrammarError has no stable import
        raise TesseraExportLaneError(
            f"tessera.export.ActivationSource refuses this capture: {exc}"
        ) from exc


def _bound_hessian_capture(hessian_path: Path) -> tuple:
    """``(hessians, provenance, capture_sha256)`` of the payload itself.

    Until RobTand/prismaquant#204 this read ``<capture>.provenance.json``
    first and touched the payload only when there was none -- but Tessera's
    ``ActivationSource.from_capture`` never reads the sidecar, so the gate
    compared the allocation against a file the encode ignores.  The identity
    is now read from the ``.pt`` the exporter loads, and its content is
    digested (:func:`hessian_capture_sha256`).  A sidecar, when present, must
    carry ``capture_sha256`` equal to that digest: the campaign writes it so
    (``tessera_campaign.write_export_inputs``), and a sidecar that seals a
    different payload -- stale, or written before the seal existed -- is
    refused by name rather than read.
    """
    import torch

    payload = torch.load(str(hessian_path), map_location="cpu",
                         weights_only=False)
    hessians = payload.get("H") if isinstance(payload, Mapping) else None
    if not isinstance(hessians, Mapping) or not all(
            hasattr(h, "detach") for h in hessians.values()):
        raise TesseraExportLaneError(
            f"--hessian {hessian_path} carries no 'H' mapping of unit -> "
            "tensor; a capture payload is {'H': {unit: [cols, cols]}, "
            "'provenance': {...}} and nothing else can be bound to the "
            "allocation")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TesseraExportLaneError(
            f"--hessian {hessian_path} carries no provenance block; a capture "
            "whose identity cannot be read cannot be bound to the allocation")
    digest = hessian_capture_sha256(hessians, provenance)
    sidecar = hessian_path.with_name(hessian_path.name + ".provenance.json")
    if sidecar.is_file():
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        sealed = loaded.get("capture_sha256") if isinstance(loaded, Mapping) \
            else None
        if sealed != digest:
            raise TesseraExportLaneError(
                f"{sidecar} does not seal the payload beside it (sidecar "
                f"capture_sha256={sealed!r}, payload={digest}). Tessera's "
                "exporter reads only the payload, so a sidecar that describes "
                "something else is a stale or pre-#204 record: re-run the "
                "campaign, which writes both files sealed together, or delete "
                "the sidecar and let the gate read the payload alone."
            )
    return hessians, provenance, digest


def _crosscheck_capture_seal(hessian_path, hessians, provenance,
                             digest) -> "str | None":
    """Refuse when Tessera's own seal of the payload is not ours; the name of
    the rule cross-checked, or None where the running Tessera has none."""
    producer = _tessera_capture_seal(hessians, provenance)
    if producer is None:
        return None
    if producer != digest:
        raise TesseraExportLaneError(
            f"--hessian {hessian_path}: PrismaQuant's hessian_capture_sha256 "
            f"({digest}) disagrees with tessera.export.ActivationSource."
            f"capture_sha256 ({producer}) for the same payload. The two "
            "digest rules have drifted and no allocation can be bound "
            "against this Tessera until they agree.")
    return "tessera.export.ActivationSource.capture_sha256"


def require_priced_export_inputs(
        assignment_path: str | Path, *, hessian_path: str | Path | None = None,
        input_scales_path: str | Path | None = None) -> dict:
    """Refuse an export whose priced inputs are not the supplied inputs.

    The allocation was priced by the campaign under two inputs the external
    exporter must be handed back, or the artifact built is not the artifact
    priced (RobTand/prismaquant#193):

    * **The Hessian.**  The encoder's shipping default is activation-aware;
      an allocation whose ``tessera_hessian`` metadata says ``supplied: true``
      names bytes shaped by a specific ``XᵀX`` capture, and the exporter's
      ``--hessian`` must carry that capture -- checked by the identity triple
      the allocation records (#195's canonical stamp) AND by the content
      digest of the payload (#204's ``capture_sha256``,
      :func:`hessian_capture_sha256`) against the ``.pt`` the exporter loads,
      never against a sidecar it does not read.  ``supplied: false`` is the
      deliberate weights-only price, and handing the exporter a Hessian then
      ships bytes the allocation never priced -- refused in that direction
      too.  An allocation that declares neither is ambiguous and fails closed
      (AGENTS.md principle 2); one that carries the triple but no digest is
      unbound and refused by name, never read as "any capture of this draw".

    * **The static activation scales.**  Every selected rung whose route
      executes the static NVFP4 contract was priced under a calibrated
      ``input_global_scale``, and the exporter refuses NVFP4 routes without
      ``--input-scales`` -- but only after encoding everything else.  This
      gate requires the file, every selected W4A4 unit's key in it, and (#204)
      that the VALUE under each key is the value the allocation's
      ``tessera_activation_static_scales`` block says priced that unit,
      before a single unit is encoded.
    """
    from .footprint import _read_safetensors_header
    from .layer_config import load_assignment, read_layer_config_metadata
    from .tessera_formats import (
        parse_tessera_format_name, tessera_serving_route, tessera_wire_recipe,
    )

    selected = {name: fmt for name, fmt in load_assignment(assignment_path).items()
                if fmt.startswith("TESSERA_")}
    report = {
        "hessian_required": False, "hessian": None,
        "hessian_capture_sha256": None, "hessian_capture_seal_crosscheck": None,
        "input_scales_required": False, "input_scales": None,
        "w4a4_units": 0, "input_scales_bound_units": 0,
    }
    if not selected:
        return report

    metadata = read_layer_config_metadata(assignment_path)
    block = metadata.get("tessera_hessian")
    if not isinstance(block, Mapping) or not isinstance(
            block.get("supplied"), bool):
        raise TesseraExportLaneError(
            "the allocation selects Tessera units but its metadata declares "
            "no tessera_hessian pricing state (supplied: true|false). Whether "
            "these bytes were priced H-aware is not recoverable from the "
            "weights, and an ambiguous allocation fails closed: re-allocate "
            "from a campaign cost table, which stamps the block."
        )
    if block["supplied"]:
        report["hessian_required"] = True
        if hessian_path is None:
            raise TesseraExportLaneError(
                "the allocation was priced H-aware (tessera_hessian.supplied "
                "= true) and no --hessian capture was supplied. The exporter "
                "would encode weights-only without refusing, shipping bytes "
                "the allocation did not price. Pass TESSERA_HESSIAN= the "
                "campaign's hessian_capture.pt (written beside its "
                "--cache-dir), or re-allocate from a --hessian off table to "
                "price weights-only deliberately."
            )
        hessian_path = Path(hessian_path)
        if not hessian_path.is_file():
            raise TesseraExportLaneError(
                f"--hessian {hessian_path} does not exist")
        expected = {field: block.get(field)
                    for field in PRICED_HESSIAN_IDENTITY_FIELDS}
        if any(value is None for value in expected.values()):
            raise TesseraExportLaneError(
                "the allocation's tessera_hessian block carries no required "
                f"identity triple ({sorted(expected)}), so no capture can be "
                "bound to it. This allocation came from a pre-triple cost "
                "table; rebuild the cost table with the current campaign and "
                "re-allocate."
            )
        priced_digest = block.get("capture_sha256")
        if not isinstance(priced_digest, str) or not priced_digest:
            raise TesseraExportLaneError(
                "the allocation's tessera_hessian block carries no "
                "capture_sha256, so the capture that priced it is unbound: "
                "the identity triple names the token draw, not the Hessian "
                "content, and two captures of one draw can encode different "
                "bytes. This allocation came from a pre-#204 cost table; "
                "re-run the campaign (its rows now carry the digest of the "
                "capture it writes) and re-allocate."
            )
        hessians, identity, digest = _bound_hessian_capture(hessian_path)
        role = identity.get("hessian_role")
        if role is not None and role != "fit":
            raise TesseraExportLaneError(
                f"--hessian {hessian_path} is a {role!r} capture and must "
                "not shape bytes")
        mismatched = {
            field: (identity.get(field), value)
            for field, value in expected.items()
            if identity.get(field) != value
        }
        if mismatched:
            raise TesseraExportLaneError(
                f"--hessian {hessian_path} is not the capture that priced "
                "this allocation: "
                + "; ".join(
                    f"{field}: capture={got!r} != allocation={want!r}"
                    for field, (got, want) in sorted(mismatched.items()))
                + ". An encode against a different Hessian ships bytes the "
                  "allocation did not price; hand the campaign's own capture "
                  "or re-allocate."
            )
        if digest != priced_digest:
            raise TesseraExportLaneError(
                f"--hessian {hessian_path} is not the capture that priced "
                f"this allocation: capture_sha256 payload={digest} != "
                f"allocation={priced_digest}. Its identity triple agrees, so "
                "this is the same token draw over different Hessian content "
                "or capture context (model, seqlen, source) -- a rewritten, "
                "re-captured or corrupted payload. An encode against it ships "
                "bytes the allocation did not price; hand the campaign's own "
                "capture or re-allocate."
            )
        report["hessian"] = str(hessian_path)
        report["hessian_capture_sha256"] = digest
        report["hessian_capture_seal_crosscheck"] = _crosscheck_capture_seal(
            hessian_path, hessians, identity, digest)
    elif hessian_path is not None:
        raise TesseraExportLaneError(
            "the allocation was priced weights-only (tessera_hessian."
            "supplied = false) but --hessian was supplied. An H-aware encode "
            "of a weights-only-priced allocation ships bytes the allocation "
            "did not price -- the same drift in the other direction. Drop "
            "the flag, or re-price with --hessian require."
        )

    w4a4 = []
    for name, fmt in sorted(selected.items()):
        parsed = parse_tessera_format_name(fmt)
        if parsed is None:
            raise TesseraExportLaneError(
                f"{name}: {fmt!r} is not a Tessera format name")
        family, rung = parsed
        wire = tessera_wire_recipe(family, rung)
        if tessera_serving_route(
                family, wire, rung).activation_source_format == "NVFP4":
            w4a4.append(name)
    report["w4a4_units"] = len(w4a4)
    if w4a4:
        report["input_scales_required"] = True
        if input_scales_path is None:
            raise TesseraExportLaneError(
                f"{len(w4a4)} selected unit(s) execute the static NVFP4 "
                "activation contract (first: " + w4a4[0] + ") and no "
                "--input-scales file was supplied. The exporter requires one "
                "input_global_scale per W4A4 module and would refuse -- after "
                "encoding everything else. Pass TESSERA_INPUT_SCALES= the "
                "campaign's input_scales.safetensors (written beside its "
                "--cache-dir), whose values are the scales the costs were "
                "priced under."
            )
        # The allocation's own record of what priced each unit comes before
        # the file: an allocation with nothing to compare against is unbound,
        # and a file that merely has the key is exactly what #204 refused to
        # keep accepting.
        priced_block = metadata.get("tessera_activation_static_scales")
        priced_units = priced_block.get("units") if isinstance(
            priced_block, Mapping) else None
        if not isinstance(priced_units, Mapping):
            raise TesseraExportLaneError(
                f"{len(w4a4)} selected unit(s) execute the static NVFP4 "
                "activation contract but the allocation's metadata carries "
                "no tessera_activation_static_scales block, so the "
                "input_global_scale each was priced under is unbound and "
                "the file's values cannot be checked. This allocation came "
                "from a pre-#204 allocator; re-allocate from the campaign's "
                "cost table, whose rows carry the priced scale."
            )
        unpriced = [name for name in w4a4
                    if not isinstance(priced_units.get(name), (int, float))
                    or isinstance(priced_units.get(name), bool)]
        if unpriced:
            raise TesseraExportLaneError(
                "the allocation priced no input_global_scale for selected "
                f"W4A4 unit(s) {unpriced[:5]}{'...' if len(unpriced) > 5 else ''}"
                " (tessera_activation_static_scales.units has no numeric "
                "value for them); a static-contract unit whose priced scale "
                "is unknown cannot be bound to any file. Re-run the campaign "
                "so every W4A4 row carries its scale, and re-allocate."
            )
        input_scales_path = Path(input_scales_path)
        if not input_scales_path.is_file():
            raise TesseraExportLaneError(
                f"--input-scales {input_scales_path} does not exist")
        header = _read_safetensors_header(str(input_scales_path))
        missing = [name for name in w4a4
                   if f"{name}.input_global_scale" not in header]
        if missing:
            raise TesseraExportLaneError(
                f"--input-scales {input_scales_path} carries no "
                "input_global_scale for selected W4A4 unit(s) "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}; the "
                "exporter's fused join cannot invent a member's scale, and a "
                "partial file exports a module the costs did not price."
            )
        from safetensors import safe_open

        with safe_open(str(input_scales_path), framework="pt",
                       device="cpu") as handle:
            for name in w4a4:
                tensor = handle.get_tensor(f"{name}.input_global_scale")
                if tensor.numel() != 1:
                    raise TesseraExportLaneError(
                        f"--input-scales {input_scales_path}: "
                        f"{name}.input_global_scale has shape "
                        f"{list(tensor.shape)}; the static NVFP4 contract "
                        "reads one scalar per unit and nothing else can be "
                        "compared with the priced scale."
                    )
                served = float(tensor.reshape(-1)[0].item())
                # The campaign prices and writes the F32-rounded scalar; the
                # comparison is between the F32 the file carries and the F32
                # of what the row says priced it, exactly.
                try:
                    priced = struct.unpack(
                        "<f", struct.pack("<f", float(priced_units[name])))[0]
                except (OverflowError, struct.error) as exc:
                    raise TesseraExportLaneError(
                        f"the allocation's priced input_global_scale for "
                        f"{name} ({priced_units[name]!r}) is not an F32 "
                        "scalar and cannot have priced any served unit"
                    ) from exc
                if served != priced:
                    raise TesseraExportLaneError(
                        f"--input-scales {input_scales_path}: "
                        f"{name}.input_global_scale = {served!r} but the "
                        f"allocation priced {priced!r} for {name}. The "
                        "served activation quantisation is a function of this "
                        "scalar, so this file exports a unit the costs did not "
                        "price; hand the campaign's own input_scales."
                        "safetensors (written beside its --cache-dir with the "
                        "capture) or re-allocate from a table priced under "
                        "this file."
                    )
        report["input_scales"] = str(input_scales_path)
        report["input_scales_bound_units"] = len(w4a4)
    return report


# ---------------------------------------------------------------------------
# The driver's entry point
# ---------------------------------------------------------------------------
def preflight(model_path: str | Path, *, target=None,
              assignment_path: str | Path | None = None,
              hessian_path: str | Path | None = None,
              input_scales_path: str | Path | None = None,
              cached_expert_units: bool = False) -> dict:
    """Every gate, in the order that puts the cheapest refusal first.

    ``cached_expert_units`` additionally writes the producer's cached-unit
    bundle for the priced expert wires this allocation selected, into the
    campaign's own wire directory, and names it in the build anchor so the
    driver hands the exporter the path rather than reconstructing it.  An
    allocation that selects no routed expert unit bundles nothing.
    """
    structure = require_declared_structure(model_path)
    target = require_serving_target(target)
    executes = require_executes_derived_from_contract()
    producer_tools = require_producer_tools()
    producer_repos = require_producer_repo_is_pinned()
    require_release_pin()
    scope = None
    build = None
    priced_inputs = None
    if assignment_path is not None:
        from .layer_config import read_layer_config_metadata
        from .shipcard import file_sha256

        assignment_sha = file_sha256(assignment_path)
        if assignment_sha is None:
            raise TesseraExportLaneError(f"cannot hash allocation {assignment_path}")
        priced_inputs = require_priced_export_inputs(
            assignment_path, hessian_path=hessian_path,
            input_scales_path=input_scales_path)
        scope = require_assignment_scope(model_path, assignment_path, target=target)
        build = {
            "source_model": str(model_path), "layer_config": str(assignment_path),
            "layer_config_sha": assignment_sha,
        }
        if scope is not None:
            build["tessera_serving_scope"] = read_layer_config_metadata(
                assignment_path)["tessera_serving_scope"]
            projection = scope.get("expert_projection")
            if projection is not None:
                build["tessera_expert_stack_formats"] = dict(projection["stacks"])
                if cached_expert_units:
                    build["cached_expert_units"] = str(
                        write_cached_expert_units(projection))
        if file_sha256(assignment_path) != assignment_sha:
            raise TesseraExportLaneError(
                "allocation changed during scoped preflight; no build anchor was produced")
    from .tessera_serving_runtime_pin import (
        load_tessera_serving_runtime_pin,
    )

    pin = load_tessera_serving_runtime_pin()
    from .lane_spec import load_lane_spec
    from .shipcard import lane_gate_slots

    spec = load_lane_spec("tessera")
    report = {
        "structure": structure,
        "executes": list(executes),
        "producer_tools": list(producer_tools),
        "producer_repos": list(producer_repos),
        "unsupported_producer_tools": [
            f"${{{tool.repo_env}}}/{tool.path} ({tool.tracking_issue})"
            for tool in spec.producer_tools if tool.stability != "supported"
        ],
        "shipcard_slots": list(lane_gate_slots("tessera")),
        "unrecorded_gates": [
            {"gate": g.id, "reason": g.unrecorded_reason}
            for g in spec.unrecorded_gates()
        ],
        "pinned_version": pin.version,
        "pinned_commit": pin.commit,
        "quant_method": "tessera",
    }
    if target is not None:
        report["serving_target"] = target.as_dict()
    if scope is not None:
        report["selected_serving_scope"] = scope
    if priced_inputs is not None:
        report["priced_inputs"] = priced_inputs
    if build is not None:
        report["build"] = build
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m prismaquant.tessera_export_lane",
        description=(
            "Preflight the EXPORT_CONTAINER=tessera arm: release pin, "
            "principle-14 executes derivation, declared structure."),
    )
    parser.add_argument("--model", required=True,
                        help="the source checkpoint run-pipeline.sh is building")
    parser.add_argument("--assignment", default=None,
                        help="selected layer_config.json to attest before plan translation")
    parser.add_argument("--hessian", default=None,
                        help="the Hessian capture the exporter will be handed "
                             "(the campaign's hessian_capture.pt); required "
                             "and identity-checked when the allocation was "
                             "priced H-aware, refused when it was not")
    parser.add_argument("--input-scales", default=None,
                        help="safetensors of <unit>.input_global_scale (the "
                             "campaign's input_scales.safetensors); required "
                             "to cover every selected W4A4 unit")
    parser.add_argument("--target-profile", default=None,
                        help="serving profile supplying or cross-checking the exact platform")
    parser.add_argument("--write-build-json", default=None,
                        help="write validated allocation facts for lane_shipcard open --build-json")
    parser.add_argument("--write-cached-expert-units", action="store_true",
                        help="bundle the priced expert wires this allocation "
                             "selected into the campaign's wire directory "
                             "(tessera.cached_units.v1) and name the manifest "
                             "in the build anchor, for the exporter's "
                             "--cached-expert-units intake")
    from .tessera_serving_scope import add_serving_scope_arguments, serving_target_from_args

    add_serving_scope_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if args.write_build_json is not None and args.assignment is None:
            raise TesseraExportLaneError("--write-build-json requires --assignment")
        if args.write_cached_expert_units and args.assignment is None:
            raise TesseraExportLaneError(
                "--write-cached-expert-units bundles the wires an ALLOCATION "
                "selected; pass --assignment")
        platform = None
        if args.target_profile is not None:
            from .serving_profiles import load_serving_profile

            platform = load_serving_profile(args.target_profile).target_platform
        target = serving_target_from_args(args, target_platform=platform)
        priced = {"hessian_path": args.hessian,
                  "input_scales_path": args.input_scales}
        if target is None and args.assignment is None:
            if args.hessian is not None or args.input_scales is not None:
                raise TesseraExportLaneError(
                    "--hessian/--input-scales bind priced inputs to an "
                    "allocation; pass --assignment")
            report = preflight(args.model)
        else:
            report = preflight(args.model, target=target,
                               assignment_path=args.assignment,
                               cached_expert_units=args.write_cached_expert_units,
                               **priced)
        if args.write_build_json is not None:
            destination = Path(args.write_build_json)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_text(json.dumps(report["build"], indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
            temporary.replace(destination)
    except (TesseraExportLaneError, OSError, ValueError) as exc:
        print(f"[preflight] ERROR: EXPORT_CONTAINER=tessera: {exc}",
              file=sys.stderr)
        return 2
    print("[preflight] tessera lane OK: "
          f"structure={report['structure']} "
          f"executes={report['executes']} "
          f"pin={report['pinned_version']}@{report['pinned_commit'][:12]}")
    print("[preflight] ship record this artifact must close: "
          + ", ".join(report["shipcard_slots"]))
    if "serving_target" in report:
        print("[preflight] explicit serving target: " + json.dumps(report["serving_target"], sort_keys=True))
    if "selected_serving_scope" in report:
        print("[preflight] scoped selected units: "
              + str(len(report["selected_serving_scope"]["by_unit"])))
    if "cached_expert_units" in report.get("build", {}):
        print("[preflight] priced expert wires handed to the exporter: "
              + report["build"]["cached_expert_units"])
    for gate in report["unrecorded_gates"]:
        print(f"[preflight] gate {gate['gate']} is ADVISORY BY DECLARATION "
              f"(closes no shipcard slot): {gate['reason']}")
    for tool in report["unsupported_producer_tools"]:
        print(f"[preflight] producer-tool debt: {tool} has no stability "
              "promise")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
