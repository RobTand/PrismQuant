"""`LaneSpec` — one lane-uniform ship gate across native / CB / GGUF.

Re-vet **R16** (`docs/audits/architecture_re-vet_2026-07-30.md`), which closes
the measurement half of debt **D26**.

**The asymmetry was wiring, not capability.** `validate_quantized_model.py` is
already runtime-agnostic — it drives an OpenAI-compatible endpoint over HTTP
(`--base-url`, `--model-name`) and knows nothing about compressed-tensors;
`grep 'gguf\\|nvfp4_cb'` across every `validate_*.py` returns zero hits. What
was missing is a *declaration* of what each lane's serve command, endpoint,
gate set and KL evaluator are, so "the bar" is defined once instead of being
native's bar by default and nothing elsewhere.

Same idiom as `serving_profile_specs/`: JSON declarations plus a dataclass with
`from_dict`, so a new lane is a data file rather than a branch.

**Gates are ADVISORY *to the build run*, and BLOCKING at publication.**
Nothing in this module fails a pipeline run — `LaneSpec.advisory_gates` is
`True` for every lane and a test pins it, so a future flip is a deliberate
edit rather than a drift. What makes a gate more than a printed sentence is
its `shipcard_slot`: the build lane OPENS the slot, the serve lane FILLS it,
and `tools/publish_artifact.py` refuses an artifact whose slots are not
closed. Whether gates should also fail the *run* is R16's open half, deferred
to Robert.

**So a gate with no slot is enforced by nothing**, and until 2026-09-03 that
was indistinguishable from a gate someone had simply not filled in.
`LaneGate.from_dict` now refuses a null `shipcard_slot` that carries no
`unrecorded_reason`, and `LaneSpec.wired_architectures` / `producer_tools` put
the lane's architecture roster and its external build-tool dependencies in the
same declaration instead of in a test constant and a bash loop. The chain a
reader should be able to follow is: gate declared here -> slot opened by
`prismaquant.lane_shipcard` in the lane's driver arm -> refused by
`publish_artifact`. Where any link is missing, the gate is a confession log
(RobTand/prismaquant#119, principle 9).
"""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "prismaquant.lane_spec.v1"

_SPEC_DIR = Path(__file__).resolve().parent / "lane_specs"


@dataclass(frozen=True)
class LaneEndpoint:
    """How a served artifact of this lane is talked to."""

    kind: str                       # openai | llama_server | none
    base_url: str | None = None
    health_path: str | None = None
    metrics_path: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LaneEndpoint":
        return cls(
            kind=str(payload["kind"]),
            base_url=_opt_str(payload.get("base_url")),
            health_path=_opt_str(payload.get("health_path")),
            metrics_path=_opt_str(payload.get("metrics_path")),
        )


@dataclass(frozen=True)
class LaneGate:
    """One numeric gate, and the shipcard slot its record closes.

    **A gate that records nothing cannot refuse anything.**  ``shipcard_slot``
    is the entire mechanism by which a declared gate becomes enforceable: the
    build lane opens the slot, the serve lane fills it, and
    ``tools/publish_artifact.py`` refuses an artifact whose slots are not
    closed.  A gate with no slot is a printed sentence -- it appears in
    ``python -m prismaquant.lane_spec``'s output and in nothing else.

    That is a legitimate state for a gate that genuinely has nowhere to record
    (a diagnostic, an operator instruction), and an illegitimate one for a
    gate an artifact's honesty depends on.  Telling the two apart has to be a
    VALUE, so ``shipcard_slot: null`` now requires ``unrecorded_reason``: a
    declaration that this gate is advisory-by-construction and why.  Omitting
    both raises at parse time, the same way ``executes`` raises rather than
    defaulting -- the Tessera lane declared ``route.census`` (principle 12's
    second leg) with a null slot and no reason for a day, which is exactly the
    "named but never run" shape RobTand/prismaquant#119 describes.
    """

    id: str
    runner: str
    shipcard_slot: str | None = None
    description: str = ""
    unrecorded_reason: str = ""

    @property
    def recorded(self) -> bool:
        """Does closing this gate leave a mark the shipcard can refuse on?"""
        return self.shipcard_slot is not None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LaneGate":
        slot = _opt_str(payload.get("shipcard_slot"))
        reason = str(payload.get("unrecorded_reason", "")).strip()
        if slot is None and not reason:
            raise ValueError(
                f"lane gate {payload.get('id')!r} declares no shipcard_slot "
                "and no `unrecorded_reason`. A gate with no slot is recorded "
                "nowhere and refuses nothing, so a lane that wants one must "
                "SAY that it is advisory by construction; silence reads as "
                "an enforced gate and is not (RobTand/prismaquant#119)")
        if slot is not None and reason:
            raise ValueError(
                f"lane gate {payload.get('id')!r} declares both a "
                f"shipcard_slot ({slot!r}) and an `unrecorded_reason`; a gate "
                "that records is not unrecorded, and carrying both leaves two "
                "readings of one gate")
        return cls(
            id=str(payload["id"]),
            runner=str(payload["runner"]),
            shipcard_slot=slot,
            description=str(payload.get("description", "")),
            unrecorded_reason=reason,
        )


@dataclass(frozen=True)
class LaneProducerTool:
    """One external tool this lane's BUILD arm shells out to.

    PrismaQuant names a serving runtime's own tools rather than vendoring
    them: a wire recipe with two homes is how the two halves of one format
    drift apart.  The cost of that boundary is a dependency on a file in
    another repository, and a dependency a reader cannot see is a dependency
    someone tidies away.  Until 2026-09-03 the Tessera arm's two were a
    hardcoded ``for`` loop in ``run-pipeline.sh`` and a sentence in this
    spec's ``notes``; neither is something a gate can read, and neither
    survives a fourth lane.

    ``stability`` is the field that matters.  ``supported`` means the tool is
    a console entry point or a public module of its package.
    ``unsupported_experiments`` means it lives under that repository's
    ``experiments/`` with no stability promise -- true today of both Tessera
    tools -- and REQUIRES ``tracking_issue``, so the debt is named on the
    artifact's own lane declaration rather than only in an issue tracker.
    """

    repo_env: str
    path: str
    stability: str
    description: str = ""
    tracking_issue: str = ""

    #: The vocabulary. A new value is a deliberate edit here, not a typo in a
    #: spec that silently reads as "supported".
    STABILITIES = ("supported", "unsupported_experiments")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LaneProducerTool":
        stability = str(payload.get("stability", "")).strip()
        if stability not in cls.STABILITIES:
            raise ValueError(
                f"lane producer tool {payload.get('path')!r} declares "
                f"stability={stability!r}; known: {list(cls.STABILITIES)}. An "
                "absent or misspelt value must not read as `supported`")
        tracking = str(payload.get("tracking_issue", "")).strip()
        if stability == "unsupported_experiments" and not tracking:
            raise ValueError(
                f"lane producer tool {payload.get('path')!r} is declared "
                "`unsupported_experiments` and names no `tracking_issue`; a "
                "shipping lane may depend on a script with no stability "
                "promise, but not silently")
        return cls(
            repo_env=str(payload["repo_env"]),
            path=str(payload["path"]),
            stability=stability,
            description=str(payload.get("description", "")),
            tracking_issue=tracking,
        )


@dataclass(frozen=True)
class LaneKLEvaluator:
    """The lane's held-out KL evaluator, behind the `validate_assignments_kl`
    interface: `(mean, per_sequence, stats)` with the gold lane's key names."""

    kind: str                       # validate_assignments_kl | llama_perplexity
    entrypoint: str
    note: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LaneKLEvaluator":
        return cls(
            kind=str(payload["kind"]),
            entrypoint=str(payload["entrypoint"]),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class LaneActivationContract:
    """Which formats' activation quantization this lane's runtime EXECUTES.

    The format registry declares NVFP4 a W4A4 format. That is a fact about the
    FORMAT, not about the LANE. Gridbook's CB runtime decodes to BF16 and runs
    a BF16 GEMM -- what its own docstring calls "the exact native BF16 bridge"
    -- unless a fused activation mode is explicitly selected by a
    process-global env selector. Every gate and gold serve on the nvfp4_cb lane
    leaves those selectors unset, so an NVFP4_CB unit's activations are never
    quantized there and its A-side cost is exactly zero.

    Pricing an A-side the runtime does not execute is a CURRENCY error, not a
    conservative overestimate. It makes a format look more expensive than it
    is, and the DP then spends real weight bytes escaping a cost of zero: on
    DSv4-Flash at 87.403 GB it promoted 2,307 units to FP8_CB and funded that
    by dropping the bulk of the model from codebook rung K16 to K12 (four fewer
    index bits on ~19k units). Discovered 2026-08-17; the same mispricing is on
    the Qwen3.8-27B CB-A allocation, which used `cost_aura_anchored_aqua.pkl`
    while serving with the same selectors unset.

    ``executes`` is the authority the A-side pricing must intersect with. It is
    a set of **glob patterns** over format names, because the answer is per
    FAMILY and the rungs within a family are open-ended: the CB lane bridges
    every ``NVFP4_CB_K*`` but genuinely serves every ``FP8_CB_K*`` as W8A8
    (`gridbook/linear.py` feeds quantized ``xq`` with per-token dynamic scales
    into ``native_cutlass_scaled_mm``; `moe.py` declares
    ``_FP8_GROUPED_CONTRACT = "fp8_per_token_dynamic"``). Listing rungs
    explicitly would silently under-declare the day a new one is added, which
    is the same silent-default failure this class exists to remove.

    An empty set is a meaningful answer -- not a missing declaration.
    """

    executes: frozenset[str]
    rationale: str
    selectors_must_be_unset: tuple[str, ...] = ()

    def matches(self, format_name: str) -> bool:
        """Does this lane execute ``format_name``'s activation quantization?"""
        return any(fnmatch.fnmatchcase(format_name, pattern)
                   for pattern in self.executes)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LaneActivationContract":
        if "executes" not in payload:
            raise ValueError(
                "served_activation_quantization must state `executes` "
                "explicitly; an absent list is not an empty list, and "
                "guessing it is the bug this field exists to prevent")
        return cls(
            executes=frozenset(str(f) for f in payload["executes"]),
            rationale=str(payload.get("rationale", "")),
            selectors_must_be_unset=tuple(
                str(s) for s in payload.get("selectors_must_be_unset", ())),
        )


@dataclass(frozen=True)
class LaneSpec:
    id: str
    export_container: str
    runtime: str
    description: str
    endpoint: LaneEndpoint
    kl_evaluator: LaneKLEvaluator
    wired_architectures: frozenset[str] = frozenset()
    serve_scripts: tuple[str, ...] = ()
    serve_command: tuple[str, ...] = ()
    gates: tuple[LaneGate, ...] = ()
    producer_tools: tuple[LaneProducerTool, ...] = ()
    serving_profiles: tuple[str, ...] = ()
    advisory_gates: bool = True
    notes: tuple[str, ...] = field(default=())
    served_activation_quantization: LaneActivationContract | None = None

    #: ``wired_architectures`` value meaning "every architecture", used by the
    #: default lane: every model profile ships through compressed-tensors, so
    #: enumerating them here would be a second roster to keep in step with
    #: ``model_profiles/registry.py``.
    ANY_ARCHITECTURE = "*"

    def wires(self, profile_name: str) -> bool:
        """Is ``profile_name`` declared wired for this lane?"""
        if self.ANY_ARCHITECTURE in self.wired_architectures:
            return True
        return profile_name in self.wired_architectures

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LaneSpec":
        schema = str(payload.get("schema", SCHEMA))
        if schema != SCHEMA:
            raise ValueError(f"unknown lane spec schema {schema!r}")
        serve = payload.get("serve", {}) or {}
        return cls(
            id=str(payload["id"]),
            export_container=str(payload["export_container"]),
            runtime=str(payload["runtime"]),
            description=str(payload.get("description", "")),
            endpoint=LaneEndpoint.from_dict(payload["endpoint"]),
            kl_evaluator=LaneKLEvaluator.from_dict(payload["kl_evaluator"]),
            serve_scripts=tuple(str(s) for s in serve.get("scripts", ())),
            serve_command=tuple(str(s) for s in serve.get("command", ())),
            gates=tuple(
                LaneGate.from_dict(g) for g in payload.get("gates", ())),
            producer_tools=tuple(
                LaneProducerTool.from_dict(t)
                for t in payload.get("producer_tools", ())),
            wired_architectures=_wired_architectures(payload),
            serving_profiles=tuple(
                str(p) for p in payload.get("serving_profiles", ())),
            advisory_gates=bool(payload.get("advisory_gates", True)),
            notes=tuple(str(n) for n in payload.get("notes", ())),
            served_activation_quantization=(
                LaneActivationContract.from_dict(
                    payload["served_activation_quantization"])
                if payload.get("served_activation_quantization") is not None
                else None),
        )

    def gate(self, gate_id: str) -> LaneGate | None:
        for g in self.gates:
            if g.id == gate_id:
                return g
        return None

    def shipcard_slots(self) -> tuple[str, ...]:
        return tuple(g.shipcard_slot for g in self.gates if g.shipcard_slot)

    def unrecorded_gates(self) -> tuple[LaneGate, ...]:
        """Gates that close no shipcard slot, each with its declared reason."""
        return tuple(g for g in self.gates if not g.recorded)

    def render_serve_command(self, **values: str) -> tuple[str, ...]:
        """Substitute `${NAME}` placeholders in the declared serve command.

        A missing placeholder is a `KeyError` naming it — a serve command with
        an unresolved `${MODEL}` is exactly the class of mistake that produces
        a container serving the wrong artifact.
        """
        from string import Template

        out: list[str] = []
        for token in self.serve_command:
            out.append(Template(token).substitute(values))
        return tuple(out)


def _wired_architectures(payload: Mapping[str, Any]) -> frozenset[str]:
    """Parse the lane's declared architecture roster.

    REQUIRED, and required to be non-empty.  The roster is a decision, not
    something derivable from the code -- but it has to live in exactly ONE
    place, and this is it.  It used to live in `tests/test_profile_export_lanes.py`
    as two module-level sets named after two specific lanes (`GGUF_WIRED`,
    `TESSERA_WIRED`), which a fourth lane would have escaped entirely: the
    test asserted two lanes by name and said nothing about any other.  Here,
    a lane declares its own roster beside everything else about that lane and
    the profile-vs-declaration property covers every lane in the vocabulary
    without a test edit.
    """
    if "wired_architectures" not in payload:
        raise ValueError(
            f"lane {payload.get('id')!r} must declare `wired_architectures`: "
            "the set of model-profile names permitted to export through it "
            f"(or [{LaneSpec.ANY_ARCHITECTURE!r}] for a lane every "
            "architecture ships through). An absent roster is not an empty "
            "one, and a lane nobody is wired for must say so with a value")
    wired = payload["wired_architectures"]
    if isinstance(wired, str) or not isinstance(wired, (list, tuple, set,
                                                        frozenset)):
        raise ValueError(
            f"lane {payload.get('id')!r}: `wired_architectures` must be a "
            f"list of model-profile names; got {type(wired).__name__}")
    names = frozenset(str(n) for n in wired)
    if not names:
        raise ValueError(
            f"lane {payload.get('id')!r}: `wired_architectures` is empty. A "
            "lane in the EXPORT_CONTAINER vocabulary that no architecture may "
            "use is a lane whose refusal happens three layers below where an "
            "operator can read it; declare the roster or retire the lane")
    return names


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def lane_spec_names() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in _SPEC_DIR.glob("*.json")))


@lru_cache(maxsize=None)
def load_lane_spec(lane_id: str) -> LaneSpec:
    path = _SPEC_DIR / f"{lane_id}.json"
    if not path.is_file():
        raise KeyError(
            f"unknown lane {lane_id!r}; known lanes: {lane_spec_names()}")
    return LaneSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def lane_spec_for_container(export_container: str) -> LaneSpec:
    """The lane declaration for an `EXPORT_CONTAINER` value."""
    want = str(export_container).strip()
    for name in lane_spec_names():
        spec = load_lane_spec(name)
        if spec.export_container == want:
            return spec
    raise KeyError(
        f"no lane spec declares export_container={want!r}; "
        f"known: {[load_lane_spec(n).export_container for n in lane_spec_names()]}"
    )


def all_lane_specs() -> tuple[LaneSpec, ...]:
    return tuple(load_lane_spec(name) for name in lane_spec_names())


def lane_gate_report(spec: LaneSpec, shipcard: Mapping[str, Any] | None = None
                     ) -> list[dict[str, Any]]:
    """Advisory status of every declared gate against a shipcard payload.

    Returns one row per gate: `{gate, runner, shipcard_slot, filled,
    advisory}`. Nothing here refuses — `python -m prismaquant.shipcard_cli verify` is the
    refusal, and this is the lane-uniform view of what it will refuse on.
    """
    slots: Mapping[str, Any] = {}
    if shipcard:
        slots = shipcard.get("slots", {}) or {}
    rows: list[dict[str, Any]] = []
    for gate in spec.gates:
        filled = (
            gate.shipcard_slot is not None
            and slots.get(gate.shipcard_slot) is not None
        )
        rows.append({
            "gate": gate.id,
            "runner": gate.runner,
            "shipcard_slot": gate.shipcard_slot,
            "filled": bool(filled),
            "recorded": bool(gate.recorded),
            "unrecorded_reason": gate.unrecorded_reason or None,
            "advisory": bool(spec.advisory_gates),
        })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Show a lane's declared ship gate")
    p.add_argument("--lane", default=None, help="lane id (default: all)")
    p.add_argument("--export-container", default=None)
    p.add_argument("--shipcard", default=None,
                   help="shipcard.json to report gate fill against")
    args = p.parse_args(argv)

    if args.export_container:
        specs = (lane_spec_for_container(args.export_container),)
    elif args.lane:
        specs = (load_lane_spec(args.lane),)
    else:
        specs = all_lane_specs()

    card = None
    if args.shipcard:
        card = json.loads(Path(args.shipcard).read_text(encoding="utf-8"))

    for spec in specs:
        print(f"[lane] {spec.id}: container={spec.export_container} "
              f"runtime={spec.runtime} endpoint={spec.endpoint.kind} "
              f"kl={spec.kl_evaluator.kind} "
              f"gates={'ADVISORY' if spec.advisory_gates else 'BLOCKING'}")
        for row in lane_gate_report(spec, card):
            if not row["recorded"]:
                state = "UNRECORDED (advisory by declaration)"
            else:
                state = "filled" if row["filled"] else "UNFILLED"
            print(f"    {row['gate']:<28} {row['runner']:<44} "
                  f"{row['shipcard_slot'] or '-':<20} {state}")
            if row["unrecorded_reason"]:
                print(f"        reason: {row['unrecorded_reason']}")
        for tool in spec.producer_tools:
            print(f"    [producer tool] ${{{tool.repo_env}}}/{tool.path} "
                  f"stability={tool.stability}"
                  + (f" tracking={tool.tracking_issue}"
                     if tool.tracking_issue else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
