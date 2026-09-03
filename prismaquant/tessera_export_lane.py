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

**Four gates, each of which fails closed on its own.**

1. :func:`require_release_pin` -- the pinned Tessera serving runtime must be an
   exact reviewed release.  It is not, today: no Tessera release tag exists
   (RobTand/tessera#17), so this refuses every run and it is the ONLY thing
   that does.  That is the honest state of the lane and it is stated where an
   operator can act on it, rather than as ``unknown export lane`` from a
   vocabulary check three layers up.
2. :func:`require_executes_derived_from_contract` -- principle 14.  The lane
   spec's ``served_activation_quantization.executes`` states what the serving
   runtime EXECUTES, so it is DERIVED from the ``formats[]`` table the runtime
   publishes and any disagreement refuses.  Two of our own spec files once
   disagreed about one runtime; the runtime was never ambiguous.
3. :func:`require_declared_structure` -- the contract declares
   ``structures: ["dense"]`` and carries no ``routed_moe`` cell, because no
   served measurement covers routed experts.  A model with routed experts is
   refused by reading that field, not by a hardcoded list here.
4. :func:`require_producer_tools` -- every external tool the lane DECLARES it
   shells out to must exist under the env var the declaration names.  This was
   a hardcoded ``for`` loop over two paths in ``run-pipeline.sh``; it is now
   read from ``lane_specs/tessera.json``'s ``producer_tools``, which also
   carries each tool's stability and its tracking issue, so a shipping lane's
   dependency on a script with no stability promise is a value a reader and a
   gate can both see (RobTand/prismaquant#119).
"""
from __future__ import annotations

import argparse
import json
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
#: It is one of ``lane_eligibility.STRUCTURES``; the contract simply does not
#: declare it.
STRUCTURE_ROUTED_MOE = "routed_moe"
STRUCTURE_DENSE = "dense"


class TesseraExportLaneError(RuntimeError):
    """The Tessera export lane refuses this run.  Always actionable."""


# ---------------------------------------------------------------------------
# Gate 1 -- the release pin
# ---------------------------------------------------------------------------
def require_release_pin() -> None:
    """Refuse unless the pinned Tessera serving runtime is a reviewed release.

    This is the same conjunct ``tessera_render.tessera_lane_attested`` ANDs
    into producer eligibility, called here so the driver's refusal and the
    allocator's refusal are the same fact rather than two.
    """
    from . import tessera_serving_runtime_pin as pin_module

    try:
        pin = pin_module.load_tessera_serving_runtime_pin()
        pin_module.require_exact_tessera_runtime_release(pin)
    except pin_module.TesseraServingRuntimePinError as exc:
        raise TesseraExportLaneError(
            f"the pinned Tessera serving runtime is not a reviewed release: "
            f"{exc}\n"
            "  There is no Tessera release tag yet (RobTand/tessera#17), and "
            "cutting one is Rob's decision, not this pipeline's. Until then "
            "the lane is declared and gated but cannot build: every Tessera "
            "rung is producer-ineligible, so an allocation cannot select one "
            "either.\n"
            "  Resolving it is ONE reviewed commit: "
            "prismaquant/tessera_runtime/tessera_serving_runtime_pin.json's "
            "commit/version/version_is_release AND the two release constants "
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
    """Refuse a checkpoint whose structure the packaged contract omits."""
    from .lane_eligibility import load_eligibility_table

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
            "  Absence is the honest state, not an oversight: no served "
            "measurement covers routed experts, so the plugin refuses a routed "
            "MoE module by name rather than degrading it, and the exporter "
            "would pass every expert through as BF16 -- an artifact that is "
            "not the allocation that justified it."
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
# The driver's entry point
# ---------------------------------------------------------------------------
def preflight(model_path: str | Path) -> dict:
    """Every gate, in the order that puts the cheapest refusal first."""
    structure = require_declared_structure(model_path)
    executes = require_executes_derived_from_contract()
    producer_tools = require_producer_tools()
    require_release_pin()
    from .tessera_serving_runtime_pin import (
        load_tessera_serving_runtime_pin,
    )

    pin = load_tessera_serving_runtime_pin()
    from .lane_spec import load_lane_spec
    from .shipcard import lane_gate_slots

    spec = load_lane_spec("tessera")
    return {
        "structure": structure,
        "executes": list(executes),
        "producer_tools": list(producer_tools),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m prismaquant.tessera_export_lane",
        description=(
            "Preflight the EXPORT_CONTAINER=tessera arm: release pin, "
            "principle-14 executes derivation, declared structure."),
    )
    parser.add_argument("--model", required=True,
                        help="the source checkpoint run-pipeline.sh is building")
    args = parser.parse_args(argv)
    try:
        report = preflight(args.model)
    except TesseraExportLaneError as exc:
        print(f"[preflight] ERROR: EXPORT_CONTAINER=tessera: {exc}",
              file=sys.stderr)
        return 2
    print("[preflight] tessera lane OK: "
          f"structure={report['structure']} "
          f"executes={report['executes']} "
          f"pin={report['pinned_version']}@{report['pinned_commit'][:12]}")
    print("[preflight] ship record this artifact must close: "
          + ", ".join(report["shipcard_slots"]))
    for gate in report["unrecorded_gates"]:
        print(f"[preflight] gate {gate['gate']} is ADVISORY BY DECLARATION "
              f"(closes no shipcard slot): {gate['reason']}")
    for tool in report["unsupported_producer_tools"]:
        print(f"[preflight] producer-tool debt: {tool} has no stability "
              "promise")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
