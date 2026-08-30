"""Discover every weight-bearing computation in a model by traversal.

This module is the discovery walker of ``docs/design/model_coverage_ledgers.md``
("walk what runs, not what's declared") and the implementation of the R5 design
contract. It exists to kill one failure class: a parameter that feeds a matmul
through a module class the pipeline's own enumeration skips, and therefore never
becomes an allocator decision. The motivating instance is DeepSeek-V4's
``attn.wo_a`` — consumed by a grouped einsum/bmm, 17.9% of decode read traffic,
shipped by omission.

The walk has one root pair and one output:

* **Root A — the module tree.** Every named parameter and buffer becomes a
  :class:`WalkNode`, enumerated with ``remove_duplicate=False`` so tied weights
  (an embedding that is also the logits projection) keep every name.
* **Root B — one traced forward.** A :class:`WeightUseInterceptor`
  (``TorchFunctionMode``) records every matmul-family call the model executes,
  together with the parameters that feed it. Each resolved
  (parameter, op) pair becomes a :class:`WalkEdge`. The trace normally runs
  under ``FakeTensorMode`` on a meta-loaded model, so intake costs no GPU and
  no weight I/O; the interceptor is agnostic to which mode hosts it
  (``execution="real"`` runs the same interceptor over a real CPU forward).

Every node must then be **claimed** by exactly one disposition —
``decide`` (enters the allocator's domain), ``pin(reason)`` (held at source
precision on purpose), or ``exclude(reason)`` (outside the artifact's scope).
Claims come from :class:`ClaimRule` lists; the model profile supplies them
(``ModelProfile.walk_claim_rules()``). A matmul-fed node with no claim
**fails the walk**, with the node named and the op cited. That is the whole
point: ``wo_a`` claimed as ``pin(probe cannot price grouped operands yet)`` on
day one is a known debt with a name; ``wo_a`` absent is what shipped.

Operand resolution
------------------
An operand maps to its parameter by **storage identity**: the walker indexes
the ``UntypedStorage`` of every named parameter and buffer, so a view or a
per-expert slice maps to its parent parameter.

Two measured facts shape the implementation (torch 2.11, 2026-08-21):

* On meta tensors ``untyped_storage().data_ptr()`` is degenerate — every
  storage reports ``0`` — so the identity key is the ``StorageImpl`` address
  (``untyped_storage()._cdata``; ``data_ptr()`` is the fallback when the
  private field is absent). This implements the contract's storage-identity
  choice; the contract's literal ``data_ptr()`` spelling cannot distinguish
  meta storages.
* Under ``FakeTensorMode``, the *output* of a view op on a non-fake parameter
  is a fresh ``FakeTensor`` whose storage does not alias the parameter's. The
  view call itself is still visible at ``__torch_function__`` level, so the
  interceptor propagates parameter identity through a fixed allowlist of
  alias and cast ops (``view``, ``__getitem__``, ``transpose``, ``to``,
  ``contiguous``, …) and records the hop chain on the edge (``via``).

An operand that resolves to nothing — not a named tensor, not a model input,
not a tensor computed during the traced forward — is **unresolved**. An
unresolved floating-point operand in a multiplicand position is a walk
failure: it means a weight this pipeline cannot name (a parameter that was
``.to()``'d or reconstructed at init time loses storage identity and lands
here, reported rather than misattributed). Additive operands (``F.linear``'s
bias, the first argument of ``addmm``/``baddbmm``) are recorded on edges with
``role="additive"`` but are exempt from both requirements: a bias is not a
GEMM multiplicand, so it is not a weight the allocator prices.

``F.scaled_dot_product_attention`` is deliberately not captured (no weights
among its operands). ``F.embedding`` produces no edge, but its weight is
recorded as consumed (:class:`EmbeddingUse`) and still requires a claim from
root A.

Honest limits
-------------
* The trace discovers what the traced forward executes. A module the trace
  never runs is still discovered by root A and dispositioned there;
  ``trace_coverage`` records which modules executed so the gap is visible,
  not silent. Container modules (``ModuleList``, ``ModuleDict``, parameter
  containers) never execute a forward by construction and are listed
  separately.
* Identity propagates through alias and cast ops only. A weight that reaches
  a matmul through arithmetic (a dequant pattern such as
  ``weight.to(x.dtype) * scale``) is an intermediate to this walker; the
  multiply's output resolves to "computed during forward" and produces no
  edge. Extending provenance through elementwise ops is deliberately out of
  scope — the walk's job is discovery, not dataflow analysis — and the
  parameter itself still requires a claim if any op consumes it directly.
* Data-dependent control flow executes shape-wise under fake tensors;
  all-expert participation on a fake-traced MoE is the desired discovery, but
  an op that requires concrete values (``.item()``, ``nonzero``) stops the
  fake trace. That is what ``execution="real"`` exists for.

Intake for a new architecture
-----------------------------
1. Meta-load the model (``with torch.device("meta"): AutoModel...``).
2. ``walk_model(model, claim_rules=profile.walk_claim_rules())``.
3. Read the failure list. Every named parameter is either a real allocator
   unit the profile must let through (``decide``), a known debt to pin with a
   reason, or an exclusion to declare. Do not silence a failure without
   writing the reason down — the reasons land on the shipcard.

The export gate
---------------
:func:`evaluate_walk_gate` projects a :class:`WalkResult` onto a STRUCTURED
verdict (:class:`WalkGateVerdict`): ``refused``, a tuple of machine-readable
``refusal_kinds``, and a provenance payload whose lists carry
``(node, op, equation, module)`` as fields a gate can read. Prose — a
:class:`WalkFailure` ``detail``, the ``refusal_reason`` — explains to humans;
nothing branches on it. ``prismaquant/run-pipeline.sh`` invokes this module's
CLI (``python3 -m prismaquant.model_walk``) immediately before every export
lane, so an unclaimed matmul-fed parameter refuses the export instead of
shipping by omission. Three properties are policy, not implementation
accident:

* An explicit override (:data:`WALK_GATE_OVERRIDE_ENV`, reason required and
  stamped) excuses **trace incompleteness only — never a claim failure**.
  Claims have a first-class mechanism (``pin(reason)`` in the profile's
  rules); a bypass around it would be the silent-green this gate exists to
  kill.
* An UNKNOWN :class:`WalkFailure` ``kind`` refuses. Today the walk emits
  ``unclaimed`` and ``unresolved``; a future category (for example the
  Tensor-Parallel one — a quantization group boundary that does not align
  with a shard boundary at TP degree N, which would land as
  ``kind == "tp_group_boundary_misaligned"``) must make even an UNUPGRADED
  gate refuse rather than pass silently.
* The decision unit is the WHOLE LOGICAL tensor. Node names are module-tree
  names of the unsharded model, so Tensor-Parallel degree cannot change the
  walk's universe, and a disposition is a property of the logical tensor:
  sharding never turns a ``pin`` into anything else. ``stored_bytes`` fields
  are **total bytes of the logical tensor** (convention recorded in the
  verdict as ``byte_accounting.convention``); per-device accounting arrives
  with TP as an additive ``shard_policy`` annotation, never as node identity.

This module imports only torch and the standard library, so it can wrap any
torch model; the prismaquant-specific claim policy lives on the model profile.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

__all__ = [
    "BYTE_POLICY_REPLICATED",
    "BYTE_POLICY_SHARDED_EVENLY",
    "Claim",
    "ClaimRule",
    "EmbeddingUse",
    "LoadedWalk",
    "SCHEMA",
    "TraceCoverage",
    "UnresolvedOperand",
    "WalkEdge",
    "WalkError",
    "WalkFailure",
    "WalkGateRefusal",
    "WalkGateVerdict",
    "WalkNode",
    "WalkProvenance",
    "WalkResult",
    "WeightUseInterceptor",
    "WALK_GATE_OVERRIDE_ENV",
    "WALK_GATE_SCHEMA",
    "claim_rules_to_json",
    "evaluate_walk_gate",
    "load_walk",
    "per_device_bytes",
    "require_walk_coverage",
    "save_walk",
    "walk_model",
]

DISPOSITIONS = ("decide", "pin", "exclude")

# ---------------------------------------------------------------------------
# Op tables
# ---------------------------------------------------------------------------

# Matmul-family capture set, keyed by callable identity (the object
# `__torch_function__` receives), with the positions of additive (non-
# multiplicand) tensor arguments. `F.scaled_dot_product_attention` is
# excluded on purpose: none of its operands are weights.
_ADDITIVE_NONE: frozenset[int] = frozenset()
_MATMUL_FUNCS: dict[Any, tuple[str, frozenset[int]]] = {
    F.linear: ("linear", frozenset({2})),           # (input, weight, bias)
    torch.matmul: ("matmul", _ADDITIVE_NONE),
    torch.Tensor.matmul: ("matmul", _ADDITIVE_NONE),
    torch.Tensor.__matmul__: ("matmul", _ADDITIVE_NONE),
    torch.Tensor.__rmatmul__: ("matmul", _ADDITIVE_NONE),
    torch.mm: ("mm", _ADDITIVE_NONE),
    torch.bmm: ("bmm", _ADDITIVE_NONE),
    torch.mv: ("mv", _ADDITIVE_NONE),
    torch.addmm: ("addmm", frozenset({0})),
    torch.addbmm: ("addbmm", frozenset({0})),
    torch.addmv: ("addmv", frozenset({0})),
    torch.baddbmm: ("baddbmm", frozenset({0})),
    torch.einsum: ("einsum", _ADDITIVE_NONE),
    torch.tensordot: ("tensordot", _ADDITIVE_NONE),
}
# Keyword spellings of additive positions.
_ADDITIVE_KWARGS: dict[str, frozenset[str]] = {
    "linear": frozenset({"bias"}),
    "addmm": frozenset({"input"}),
    "addbmm": frozenset({"input"}),
    "addmv": frozenset({"input"}),
    "baddbmm": frozenset({"input"}),
}

_EMBEDDING_FUNCS = {F.embedding, F.embedding_bag}

# Ops through which parameter identity propagates: the output *is* the
# parameter's payload under an alias, a reshape, or a value-preserving cast.
# Matched by `__name__` because most are bound methods with many callable
# spellings. Arithmetic is deliberately absent (see "Honest limits").
_ALIAS_OP_NAMES = frozenset({
    "__getitem__", "alias", "as_strided", "bfloat16", "chunk", "clone",
    "contiguous", "detach", "double", "expand", "expand_as", "flatten",
    "float", "half", "movedim", "moveaxis", "narrow", "permute", "ravel",
    "reshape", "select", "split", "squeeze", "swapaxes", "swapdims", "t",
    "to", "transpose", "type", "type_as", "unbind", "unflatten", "unsqueeze",
    "view", "view_as",
})

_CONTAINER_CLASSES = (
    nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict,
)

# ---------------------------------------------------------------------------
# Artifact identity + the TP byte seam
# ---------------------------------------------------------------------------

#: Identity of the serialized walk artifact written by :func:`save_walk`.
#: ``load_walk`` refuses any other value (parse-time refusal, same pattern as
#: ``decision_units.parse_payload``). The gate verdict carries its own
#: sibling schema (:data:`WALK_GATE_SCHEMA`).
SCHEMA = "prismaquant.model_walk.v1"

#: Byte policies accepted by :func:`per_device_bytes`. They are the ONLY two
#: honest readings of a logical-total under a Tensor-Parallel degree: a
#: replicated tensor is read/held whole on every rank; an evenly sharded one
#: divides exactly. There is deliberately no default policy and no inferred
#: replication class yet — nothing in current code can populate one honestly,
#: and a speculative enum invites silent defaulting.
BYTE_POLICY_REPLICATED = "replicated"
BYTE_POLICY_SHARDED_EVENLY = "sharded_evenly"
BYTE_POLICIES = (BYTE_POLICY_REPLICATED, BYTE_POLICY_SHARDED_EVENLY)


def per_device_bytes(total: int, tp_degree: int, policy: str) -> int:
    """Device-local bytes for a logical total under a TP degree and policy.

    This is THE seam the Tensor-Parallel campaign builds on. Every
    ``stored_bytes`` field in this module is LOGICAL-TOTAL bytes of the whole
    parameter; per-device accounting exists only through this explicit-policy
    accessor, so a per-device reading can never be silently conflated with
    the total. ``tp_degree=1`` is the identity for every policy. An evenly
    sharded total that does not divide by the degree raises — a non-dividing
    shard boundary is exactly the misalignment class that must be loud
    (cf. the future ``tp_group_boundary_misaligned`` walk-failure kind).
    """
    if tp_degree < 1:
        raise ValueError(f"tp_degree must be >= 1, got {tp_degree}")
    if policy not in BYTE_POLICIES:
        raise ValueError(
            f"policy must be one of {list(BYTE_POLICIES)}, got {policy!r}; "
            "there is deliberately no default — say which reading you want")
    total = int(total)
    if tp_degree == 1:
        return total
    if policy == BYTE_POLICY_REPLICATED:
        return total
    if total % tp_degree:
        raise ValueError(
            f"logical total {total} does not divide by tp_degree={tp_degree} "
            f"under policy {policy!r}; a shard boundary would cross a tensor "
            "granularity boundary — refuse rather than round")
    return total // tp_degree


def _storage_key(tensor: torch.Tensor) -> int | None:
    """Identity key of a tensor's underlying storage.

    Uses the ``StorageImpl`` address (``_cdata``): views and slices of one
    parameter share it, distinct parameters differ, and — unlike
    ``data_ptr()``, which is 0 for every meta storage — it stays unique on
    the meta device. Falls back to ``data_ptr()`` if the private field ever
    disappears; returns None for tensors without accessible storage.
    """
    try:
        storage = tensor.untyped_storage()
    except Exception:
        return None
    cdata = getattr(storage, "_cdata", None)
    if cdata is not None:
        return int(cdata)
    return int(storage.data_ptr())


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


# ---------------------------------------------------------------------------
# Output dataclasses (all JSON-serializable through `to_json_dict`)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WalkNode:
    """One named parameter or buffer from the module tree (root A)."""

    name: str
    kind: str                       # "parameter" | "buffer"
    persistent: bool                # False only for non-persistent buffers
    shape: tuple[int, ...]
    dtype: str
    stored_bytes: int               # LOGICAL-TOTAL bytes of the whole tensor;
                                    # per-device readings go through
                                    # per_device_bytes(total, degree, policy)
    owner_module: str               # qualified name of the owning module
    module_class: str               # class name of the owning module
    module_class_mro: tuple[str, ...]  # class names, most-derived first
    aliases: tuple[str, ...]        # every OTHER name sharing this storage

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["shape"] = list(self.shape)
        d["module_class_mro"] = list(self.module_class_mro)
        d["aliases"] = list(self.aliases)
        return d


@dataclasses.dataclass(frozen=True)
class WalkEdge:
    """One (parameter, matmul-family op) consumption discovered by the trace
    (root B)."""

    param: str                      # primary node name
    param_aliases: tuple[str, ...]  # tied names sharing the storage
    op: str                         # "linear" | "matmul" | "einsum" | ...
    equation: str | None            # einsum equation, when the op has one
    role: str                       # "multiplicand" | "additive"
    operand_index: int              # position among the op's tensor operands
    operand_shape: tuple[int, ...]  # shape as consumed (view/slice shape)
    operand_dtype: str
    operand_shapes: tuple[tuple[int, ...], ...]  # every tensor operand's shape
    stored_bytes: int               # LOGICAL-TOTAL bytes of the full parameter
                                    # node (see WalkNode.stored_bytes)
    module: str                     # module executing the op ("" = root)
    via: tuple[str, ...]            # alias/cast hops from parameter to operand
    calls: int = 1                  # identical consumptions in the trace

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["param_aliases"] = list(self.param_aliases)
        d["operand_shape"] = list(self.operand_shape)
        d["operand_shapes"] = [list(s) for s in self.operand_shapes]
        d["via"] = list(self.via)
        return d


@dataclasses.dataclass(frozen=True)
class EmbeddingUse:
    """An `F.embedding` consumption: no edge, but the weight still requires a
    claim from root A."""

    param: str
    param_aliases: tuple[str, ...]
    module: str

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["param_aliases"] = list(self.param_aliases)
        return d


@dataclasses.dataclass(frozen=True)
class UnresolvedOperand:
    """A matmul operand that resolves to nothing the walk can name."""

    op: str
    equation: str | None
    module: str
    operand_index: int
    operand_shape: tuple[int, ...]
    operand_dtype: str
    role: str
    is_floating: bool

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["operand_shape"] = list(self.operand_shape)
        return d


@dataclasses.dataclass(frozen=True)
class Claim:
    """The disposition of one node, with the reason it carries."""

    disposition: str                # "decide" | "pin" | "exclude"
    reason: str
    rule_index: int                 # index into the applied rule list

    def to_json_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class WalkFailure:
    """One reason the walk fails: an unclaimed matmul-fed node, or an
    unresolved floating multiplicand."""

    kind: str                       # "unclaimed" | "unresolved"
    node: str | None                # the parameter/buffer name, when known
    op: str
    equation: str | None
    module: str
    detail: str

    def to_json_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TraceCoverage:
    """Which modules the traced forward executed. `containers` never execute
    a forward by construction (ModuleList and friends) and are listed apart
    so `not_executed` means what it says."""

    executed: tuple[str, ...]
    not_executed: tuple[str, ...]
    containers: tuple[str, ...]

    def to_json_dict(self) -> dict:
        return {
            "executed": list(self.executed),
            "not_executed": list(self.not_executed),
            "containers": list(self.containers),
        }


@dataclasses.dataclass(frozen=True)
class WalkResult:
    """The single enumeration every consumer derives from."""

    nodes: tuple[WalkNode, ...]
    edges: tuple[WalkEdge, ...]
    claims: dict[str, Claim]
    unclaimed: tuple[str, ...]      # nodes with no claim (fatal only if fed)
    embedding_uses: tuple[EmbeddingUse, ...]
    unresolved_operands: tuple[UnresolvedOperand, ...]
    failures: tuple[WalkFailure, ...]
    trace_coverage: TraceCoverage
    execution: str                  # "fake" | "real"
    #: Captured at trace time by :func:`walk_model`. Deliberately NOT
    #: serialized by :meth:`to_json_dict` — its timestamps would break the
    #: run-to-run byte-determinism the conformance tests pin; it travels in
    #: :func:`save_walk`'s envelope instead.
    provenance: "WalkProvenance | None" = None

    @property
    def ok(self) -> bool:
        return not self.failures

    def node(self, name: str) -> WalkNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def edges_for(self, name: str) -> tuple[WalkEdge, ...]:
        return tuple(e for e in self.edges
                     if e.param == name or name in e.param_aliases)

    def raise_if_failed(self) -> None:
        if self.failures:
            raise WalkError(self)

    def to_json_dict(self) -> dict:
        return {
            "execution": self.execution,
            "ok": self.ok,
            "nodes": [n.to_json_dict() for n in self.nodes],
            "edges": [e.to_json_dict() for e in self.edges],
            "claims": {k: v.to_json_dict()
                       for k, v in sorted(self.claims.items())},
            "unclaimed": list(self.unclaimed),
            "embedding_uses": [u.to_json_dict() for u in self.embedding_uses],
            "unresolved_operands": [
                u.to_json_dict() for u in self.unresolved_operands],
            "failures": [f.to_json_dict() for f in self.failures],
            "trace_coverage": self.trace_coverage.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "WalkResult":
        """Rehydrate the inner result payload of a saved walk artifact.

        Provenance is NOT part of this payload (see the field comment); it is
        reloaded from the envelope by :func:`load_walk` and attached there.
        """
        nodes = tuple(
            WalkNode(
                name=n["name"], kind=n["kind"], persistent=n["persistent"],
                shape=tuple(n["shape"]), dtype=n["dtype"],
                stored_bytes=int(n["stored_bytes"]),
                owner_module=n["owner_module"],
                module_class=n["module_class"],
                module_class_mro=tuple(n["module_class_mro"]),
                aliases=tuple(n["aliases"]),
            )
            for n in d["nodes"]
        )
        edges = tuple(
            WalkEdge(
                param=e["param"], param_aliases=tuple(e["param_aliases"]),
                op=e["op"], equation=e.get("equation"), role=e["role"],
                operand_index=int(e["operand_index"]),
                operand_shape=tuple(e["operand_shape"]),
                operand_dtype=e["operand_dtype"],
                operand_shapes=tuple(tuple(s) for s in e["operand_shapes"]),
                stored_bytes=int(e["stored_bytes"]), module=e["module"],
                via=tuple(e.get("via", ())), calls=int(e.get("calls", 1)),
            )
            for e in d["edges"]
        )
        claims = {
            k: Claim(
                disposition=v["disposition"], reason=v["reason"],
                rule_index=int(v["rule_index"]),
            )
            for k, v in d["claims"].items()
        }
        embedding_uses = tuple(
            EmbeddingUse(
                param=u["param"],
                param_aliases=tuple(u["param_aliases"]),
                module=u["module"],
            )
            for u in d["embedding_uses"]
        )
        unresolved = tuple(
            UnresolvedOperand(
                op=u["op"], equation=u.get("equation"), module=u["module"],
                operand_index=int(u["operand_index"]),
                operand_shape=tuple(u["operand_shape"]),
                operand_dtype=u["operand_dtype"], role=u["role"],
                is_floating=bool(u["is_floating"]),
            )
            for u in d["unresolved_operands"]
        )
        failures = tuple(
            WalkFailure(
                kind=f["kind"], node=f.get("node"), op=f["op"],
                equation=f.get("equation"), module=f["module"],
                detail=f.get("detail", ""),
            )
            for f in d["failures"]
        )
        coverage = d["trace_coverage"]
        return cls(
            nodes=nodes,
            edges=edges,
            claims=claims,
            unclaimed=tuple(d["unclaimed"]),
            embedding_uses=embedding_uses,
            unresolved_operands=unresolved,
            failures=failures,
            trace_coverage=TraceCoverage(
                executed=tuple(coverage["executed"]),
                not_executed=tuple(coverage["not_executed"]),
                containers=tuple(coverage["containers"]),
            ),
            execution=str(d["execution"]),
        )


class WalkError(RuntimeError):
    """The walk failed. The message names every failing node and cites the op
    that fed it; `result` carries the full enumeration for diagnosis."""

    def __init__(self, result: WalkResult):
        self.result = result
        lines = ["model walk failed:"]
        for f in result.failures:
            where = f"{f.op}" + (f" '{f.equation}'" if f.equation else "")
            lines.append(
                f"  [{f.kind}] {f.node or '<unnamed tensor>'} fed to {where} "
                f"in module '{f.module or '<root>'}': {f.detail}"
            )
        lines.append(
            "Every matmul-fed parameter needs a claim: decide, pin(reason), "
            "or exclude(reason) — see ModelProfile.walk_claim_rules()."
        )
        super().__init__("\n".join(lines))


# ---------------------------------------------------------------------------
# Claim rules
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ClaimRule:
    """One declarative claim: the first rule that matches a node claims it.

    Every populated matcher must hold for the rule to match. `module_class`
    matches anywhere in the owning module's MRO, so `"Linear"` claims
    subclasses too. `predicate` exists for policies a pattern cannot express
    (a profile's `is_pinned_name`); it receives the :class:`WalkNode`.
    """

    disposition: str
    reason: str
    name_regex: str | None = None
    leaf: str | None = None          # last dotted component of the node name
    module_class: str | None = None  # class name anywhere in the owner's MRO
    kind: str | None = None          # "parameter" | "buffer"
    persistent: bool | None = None   # match only (non-)persistent nodes
    max_ndim: int | None = None
    min_ndim: int | None = None
    floating: bool | None = None     # match only (non-)floating dtypes
    predicate: Callable[[WalkNode], bool] | None = None

    def __post_init__(self):
        if self.disposition not in DISPOSITIONS:
            raise ValueError(
                f"disposition must be one of {DISPOSITIONS}, "
                f"got {self.disposition!r}")
        if not self.reason:
            raise ValueError("a ClaimRule must carry a reason")

    def matches(self, node: WalkNode) -> bool:
        if self.name_regex is not None and not re.search(
                self.name_regex, node.name):
            return False
        if self.leaf is not None and node.name.rsplit(".", 1)[-1] != self.leaf:
            return False
        if (self.module_class is not None
                and self.module_class not in node.module_class_mro):
            return False
        if self.kind is not None and node.kind != self.kind:
            return False
        if self.persistent is not None and node.persistent is not self.persistent:
            return False
        ndim = len(node.shape)
        if self.max_ndim is not None and ndim > self.max_ndim:
            return False
        if self.min_ndim is not None and ndim < self.min_ndim:
            return False
        if self.floating is not None:
            if _dtype_is_floating(node.dtype) is not self.floating:
                return False
        if self.predicate is not None and not self.predicate(node):
            return False
        return True


def _dtype_is_floating(dtype_str: str) -> bool:
    dtype = getattr(torch, dtype_str.removeprefix("torch."), None)
    return isinstance(dtype, torch.dtype) and (
        dtype.is_floating_point or dtype.is_complex)


def apply_claim_rules(
    nodes: Sequence[WalkNode], rules: Sequence[ClaimRule],
) -> dict[str, Claim]:
    claims: dict[str, Claim] = {}
    for node in nodes:
        for index, rule in enumerate(rules):
            if rule.matches(node):
                claims[node.name] = Claim(
                    disposition=rule.disposition,
                    reason=rule.reason,
                    rule_index=index,
                )
                break
    return claims


# ---------------------------------------------------------------------------
# Artifact provenance (captured at trace time) + applied-rule serialization
#
# `Claim.rule_index` points into the rule list that was applied, but only the
# resolved claims used to survive serialization — so a reloaded artifact could
# not audit WHY a node is pinned beyond its reason string. The envelope below
# carries the rules themselves (matchers JSON-native; `predicate` recorded by
# name/identity, never dropped silently) plus everything a cached walk must be
# able to prove about what produced it. All of it is captured AT TRACE TIME;
# none of it is reconstructible afterward.
# ---------------------------------------------------------------------------


def _callable_identity(fn: Callable | None) -> dict | None:
    """Name/identity marker for a ClaimRule predicate.

    Predicates are code, not data: the artifact records WHAT ran (name,
    qualname, module, owner class, and for lambdas the defining source
    line) so an auditor can find it — never a pickle. The digest over the
    serialized list still changes if a different predicate object is swapped
    in, because the identity strings change with it.
    """
    if fn is None:
        return None
    owner = getattr(fn, "__self__", None)
    identity = {
        "name": getattr(fn, "__name__", "") or repr(fn),
        "qualname": getattr(fn, "__qualname__", "") or "",
        "module": getattr(fn, "__module__", "") or "",
        "owner": type(owner).__name__ if owner is not None else "",
        "location": "",
    }
    code = getattr(fn, "__code__", None)
    if code is not None and identity["name"] == "<lambda>":
        identity["location"] = f"{code.co_filename}:{code.co_firstlineno}"
    return identity


def claim_rules_to_json(rules: Sequence[ClaimRule]) -> tuple[dict, ...]:
    """Serialize an applied :class:`ClaimRule` list for the artifact.

    ``predicate`` is the one non-JSON-native matcher field; it is recorded by
    name/identity rather than dropped. The sha256 over this tuple's canonical
    JSON is :attr:`WalkProvenance.claim_rules_digest`.
    """
    out: list[dict] = []
    for index, rule in enumerate(rules):
        out.append({
            "index": index,
            "disposition": rule.disposition,
            "reason": rule.reason,
            "name_regex": rule.name_regex,
            "leaf": rule.leaf,
            "module_class": rule.module_class,
            "kind": rule.kind,
            "persistent": rule.persistent,
            "max_ndim": rule.max_ndim,
            "min_ndim": rule.min_ndim,
            "floating": rule.floating,
            "predicate": _callable_identity(rule.predicate),
        })
    return tuple(out)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _claim_rules_digest(rules: Sequence[ClaimRule]) -> str:
    return hashlib.sha256(
        _canonical(list(claim_rules_to_json(rules))).encode("utf-8")
    ).hexdigest()


@dataclasses.dataclass(frozen=True)
class WalkProvenance:
    """Everything a reloaded walk artifact must prove about its origin.

    Captured at trace time by :func:`walk_model`. Load-bearing, not
    decoration: two walks of the same weights under different input contracts
    can differ in ``trace_coverage`` when data-dependent control flow
    executes shape-wise under fake tensors, so the example-input spec travels
    with the result and ``load_walk`` refuses mismatches.
    """

    created_utc: str                    # ISO 8601, UTC
    model_identity: str                 # architecture + config-content digest
    torch_version: str
    transformers_version: str | None
    prismaquant_version: str            # "" when not installed as a dist
    prismaquant_git: str                # git describe; "" when unavailable
    execution: str                      # mirrors WalkResult.execution
    example_inputs_spec: str
    seq_len: int | None                 # set only for the synthesized default
    claim_rules_digest: str             # sha256 over `claim_rules`
    claim_rules: tuple[dict, ...]       # claim_rules_to_json() output

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["claim_rules"] = list(self.claim_rules)
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "WalkProvenance":
        return cls(
            created_utc=str(d["created_utc"]),
            model_identity=str(d["model_identity"]),
            torch_version=str(d["torch_version"]),
            transformers_version=(
                None if d.get("transformers_version") is None
                else str(d["transformers_version"])),
            prismaquant_version=str(d.get("prismaquant_version", "")),
            prismaquant_git=str(d.get("prismaquant_git", "")),
            execution=str(d["execution"]),
            example_inputs_spec=str(d["example_inputs_spec"]),
            seq_len=(None if d.get("seq_len") is None else int(d["seq_len"])),
            claim_rules_digest=str(d["claim_rules_digest"]),
            claim_rules=tuple(dict(r) for r in d.get("claim_rules", ())),
        )

    def rules_digest_matches(self, rules: Sequence[ClaimRule]) -> bool:
        return self.claim_rules_digest == _claim_rules_digest(rules)


def _model_identity(model: nn.Module) -> str:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        try:
            blob = _canonical(cfg.to_dict())
            model_type = getattr(cfg, "model_type", "") or ""
            digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
            return (
                f"{model_type or type(model).__name__}:sha256:{digest}"
            )
        except Exception:
            pass
    try:
        n_params = sum(int(p.numel()) for p in model.parameters())
    except Exception:
        n_params = -1
    return f"{type(model).__name__}:params:{n_params}"


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except (PackageNotFoundError, Exception):  # noqa: BLE001 - best effort
        return ""


def _transformers_version() -> str | None:
    try:
        import transformers

        return str(getattr(transformers, "__version__", "") or "")
    except Exception:  # noqa: BLE001 - optional dependency of the walker
        return None


def _git_describe() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).resolve().parents[1],
            check=True, text=True, timeout=10,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance degrades to "", never fails
        return ""


def _example_inputs_spec(example_inputs, seq_len: int,
                         used_default: bool) -> str:
    if used_default:
        return f"default:input_ids(1,{seq_len})+use_cache_if_accepted"
    parts: list[str] = []
    if isinstance(example_inputs, Mapping):
        items = sorted(example_inputs.items(), key=lambda kv: str(kv[0]))
        for key, value in items:
            for tensor in _iter_tensors(value):
                parts.append(f"{key}:{list(tensor.shape)}@{tensor.dtype}")
    elif isinstance(example_inputs, tuple):
        for i, value in enumerate(example_inputs):
            for tensor in _iter_tensors(value):
                parts.append(f"[{i}]:{list(tensor.shape)}@{tensor.dtype}")
    else:
        for tensor in _iter_tensors(example_inputs):
            parts.append(f"in:{list(tensor.shape)}@{tensor.dtype}")
    blob = ";".join(parts)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return f"provided:{digest}:{blob[:160]}"


def capture_walk_provenance(
    model: nn.Module,
    *,
    execution: str,
    example_inputs,
    seq_len: int | None,
    claim_rules: Sequence[ClaimRule],
    used_default_inputs: bool = False,
) -> WalkProvenance:
    """Snapshot at trace time what no later reader can reconstruct."""
    import datetime

    return WalkProvenance(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        model_identity=_model_identity(model),
        torch_version=str(torch.__version__),
        transformers_version=_transformers_version(),
        prismaquant_version=_package_version("prismaquant"),
        prismaquant_git=_git_describe(),
        execution=execution,
        example_inputs_spec=_example_inputs_spec(
            example_inputs, seq_len or 0, used_default_inputs),
        seq_len=None if seq_len is None else int(seq_len),
        claim_rules_digest=_claim_rules_digest(claim_rules),
        claim_rules=claim_rules_to_json(claim_rules),
    )


# ---------------------------------------------------------------------------
# Artifact save/load (the cache: one walk per model/config/input-contract,
# consumed many times). The envelope WRAPS today's payload — result bytes stay
# exactly what the determinism ratchet pins; timestamps live only in
# provenance.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LoadedWalk:
    """One reloaded walk artifact: the result plus its trace-time proof."""

    path: str
    result: WalkResult
    provenance: WalkProvenance


def save_walk(
    result: WalkResult,
    path: str | os.PathLike,
    *,
    provenance: WalkProvenance,
) -> Path:
    """Atomically write ``{schema, provenance, result}`` as one JSON document.

    Refuses a provenance that contradicts the one already attached to
    ``result``. Plain JSON on purpose: size at DSv4 scale is unmeasured, so
    no gzip claim is made here — measure before optimizing.
    """
    if result.provenance is not None and result.provenance != provenance:
        raise ValueError(
            "save_walk: provenance argument disagrees with the provenance "
            "captured on this WalkResult")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "provenance": provenance.to_json_dict(),
        "result": result.to_json_dict(),
    }
    blob = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode(
        "utf-8")
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, target)
    return target


def load_walk(
    path: str | os.PathLike,
    *,
    expect_claim_rules: Sequence[ClaimRule] | None = None,
) -> LoadedWalk:
    """Load a saved walk artifact, refusing anything it cannot prove.

    Fail-closed on: a foreign schema (parse-time refusal, same pattern as
    ``decision_units.parse_payload``); a provenance whose ``execution``
    disagrees with the result's own record; and — when the caller passes
    ``expect_claim_rules`` — a claim-rules digest that does not match the
    rules the caller is about to trust. A mismatch means the cached claims
    were written by a different policy and must not be served silently.
    """
    artifact = Path(path)
    payload = json.loads(artifact.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        found = payload.get("schema") if isinstance(payload, dict) else None
        raise ValueError(
            f"unsupported model-walk schema at {artifact}: {found!r} "
            f"(expected {SCHEMA!r})")
    provenance = WalkProvenance.from_json_dict(payload["provenance"])
    result = WalkResult.from_json_dict(payload["result"])
    result = dataclasses.replace(result, provenance=provenance)
    if provenance.execution != result.execution:
        raise ValueError(
            f"{artifact}: provenance execution {provenance.execution!r} "
            f"disagrees with the result's own {result.execution!r}")
    if expect_claim_rules is not None and \
            not provenance.rules_digest_matches(expect_claim_rules):
        raise ValueError(
            f"{artifact}: produced under different claim rules (digest "
            f"{provenance.claim_rules_digest[:12]}); refusing to serve "
            "claims this caller did not write")
    return LoadedWalk(path=str(artifact), result=result, provenance=provenance)


# ---------------------------------------------------------------------------
# The interceptor (root B)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _OpRecord:
    op: str
    equation: str | None
    module: str
    operands: list  # per tensor operand: (key, shape, dtype, role)


class WeightUseInterceptor(TorchFunctionMode):
    """Records every matmul-family call and resolves its operands to named
    tensors by storage identity.

    The interceptor is host-mode agnostic: it works identically over a
    ``FakeTensorMode`` trace of a meta-loaded model and over a real CPU
    forward, because it reads operands at ``__torch_function__`` level —
    before any dispatch-level fake conversion replaces them.

    ``origin`` maps a storage key to (names, via-chain); it is seeded with
    the model's named tensors and extended through alias/cast ops during the
    trace. Every tensor whose key enters ``origin`` or ``computed`` is
    appended to ``_keepalive`` for the duration of the trace, so a freed
    storage's address can never be reused by a later tensor and
    misattributed — the one failure mode storage-identity resolution must
    never have.
    """

    def __init__(self, named_storages: Mapping[int, tuple[str, ...]]):
        super().__init__()
        self.origin: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {
            key: (names, ()) for key, names in named_storages.items()
        }
        self.computed: set[int] = set()
        self.records: list[_OpRecord] = []
        self.embedding_records: list[tuple[tuple[str, ...] | None, str]] = []
        self.module_stack: list[str] = []
        self._keepalive: list[torch.Tensor] = []

    # -- module attribution (driven by the walker's forward hooks) ---------
    @property
    def current_module(self) -> str:
        return self.module_stack[-1] if self.module_stack else ""

    # -- torch function protocol ------------------------------------------
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        matmul = _MATMUL_FUNCS.get(func)
        if matmul is not None:
            self._record_matmul(matmul[0], matmul[1], args, kwargs)
        elif func in _EMBEDDING_FUNCS:
            self._record_embedding(args, kwargs)

        name = getattr(func, "__name__", "")
        if name in _ALIAS_OP_NAMES:
            self._propagate_alias(name, args, out)

        for tensor in _iter_tensors(out):
            key = _storage_key(tensor)
            if key is not None and key not in self.origin:
                if key not in self.computed:
                    self.computed.add(key)
                    self._keepalive.append(tensor)
        return out

    def mark_inputs(self, tensors: Iterable[torch.Tensor]) -> None:
        """Register the forward's inputs so they resolve as activations."""
        for tensor in tensors:
            key = _storage_key(tensor)
            if key is not None and key not in self.origin:
                self.computed.add(key)
                self._keepalive.append(tensor)

    # -- internals ---------------------------------------------------------
    def _record_matmul(self, op: str, additive_positions: frozenset[int],
                       args: tuple, kwargs: dict) -> None:
        equation = None
        tensors: list[tuple[torch.Tensor, str]] = []
        position = 0
        for arg in args:
            if isinstance(arg, str) and op == "einsum" and equation is None:
                equation = arg
                continue
            for tensor in _iter_tensors(arg):
                role = ("additive" if position in additive_positions
                        else "multiplicand")
                tensors.append((tensor, role))
                position += 1
        additive_kwargs = _ADDITIVE_KWARGS.get(op, frozenset())
        for key_name, value in kwargs.items():
            for tensor in _iter_tensors(value):
                role = ("additive" if key_name in additive_kwargs
                        else "multiplicand")
                tensors.append((tensor, role))
        record = _OpRecord(
            op=op, equation=equation, module=self.current_module, operands=[])
        for tensor, role in tensors:
            record.operands.append((
                _storage_key(tensor),
                tuple(tensor.shape),
                str(tensor.dtype),
                role,
                tensor.dtype.is_floating_point or tensor.dtype.is_complex,
            ))
        self.records.append(record)

    def _record_embedding(self, args: tuple, kwargs: dict) -> None:
        # F.embedding(input, weight, ...); F.embedding_bag(input, weight, ...)
        weight = kwargs.get("weight")
        if weight is None:
            tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
            weight = tensor_args[1] if len(tensor_args) > 1 else None
        names = None
        if isinstance(weight, torch.Tensor):
            key = _storage_key(weight)
            entry = self.origin.get(key) if key is not None else None
            if entry is not None:
                names = entry[0]
        self.embedding_records.append((names, self.current_module))

    def _propagate_alias(self, op_name: str, args: tuple, out: Any) -> None:
        source = args[0] if args and isinstance(args[0], torch.Tensor) else None
        if source is None:
            return
        key = _storage_key(source)
        entry = self.origin.get(key) if key is not None else None
        if entry is None:
            return
        names, via = entry
        hop = via if via and via[-1] == op_name else via + (op_name,)
        for tensor in _iter_tensors(out):
            out_key = _storage_key(tensor)
            if out_key is not None and out_key not in self.origin:
                self.origin[out_key] = (names, hop)
                self._keepalive.append(tensor)

    def release(self) -> None:
        """Drop the keepalive references once the trace is consumed."""
        self._keepalive.clear()


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def _named_tensor_index(model: nn.Module):
    """Root A: every named parameter and buffer, tied names preserved."""
    modules = dict(model.named_modules())
    non_persistent: set[str] = set()
    for qname, module in modules.items():
        for leaf in getattr(module, "_non_persistent_buffers_set", ()):
            non_persistent.add(f"{qname}.{leaf}" if qname else leaf)

    entries: list[tuple[str, str, torch.Tensor]] = []
    for name, param in model.named_parameters(remove_duplicate=False):
        entries.append((name, "parameter", param))
    for name, buf in model.named_buffers(remove_duplicate=False):
        entries.append((name, "buffer", buf))

    by_key: dict[int, list[str]] = {}
    for name, _, tensor in entries:
        key = _storage_key(tensor)
        if key is not None:
            by_key.setdefault(key, []).append(name)

    nodes: list[WalkNode] = []
    for name, kind, tensor in entries:
        owner = name.rsplit(".", 1)[0] if "." in name else ""
        module = modules.get(owner)
        mro: tuple[str, ...] = ()
        if module is not None:
            mro = tuple(
                cls.__name__ for cls in type(module).__mro__
                if cls not in (object,))
        key = _storage_key(tensor)
        aliases = tuple(n for n in by_key.get(key, []) if n != name)
        nodes.append(WalkNode(
            name=name,
            kind=kind,
            persistent=(kind == "parameter" or name not in non_persistent),
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            stored_bytes=tensor.numel() * tensor.element_size(),
            owner_module=owner,
            module_class=type(module).__name__ if module is not None else "",
            module_class_mro=mro,
            aliases=aliases,
        ))
    named_storages = {key: tuple(names) for key, names in by_key.items()}
    return nodes, named_storages


def _model_device(model: nn.Module) -> torch.device:
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def _default_example_inputs(model: nn.Module, seq_len: int,
                            device: torch.device) -> dict:
    """A language-model default: token ids, cache off. Callers with other
    input contracts pass `example_inputs` explicitly."""
    ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    return {"input_ids": ids}


def _call_forward(model: nn.Module, example_inputs) -> None:
    if isinstance(example_inputs, Mapping):
        kwargs = dict(example_inputs)
        if "use_cache" not in kwargs and _forward_accepts(model, "use_cache"):
            kwargs["use_cache"] = False
        model(**kwargs)
    elif isinstance(example_inputs, tuple):
        model(*example_inputs)
    else:
        model(example_inputs)


def _forward_accepts(model: nn.Module, name: str) -> bool:
    """Signature check, not try-and-retry: a retry after a TypeError raised
    mid-forward would trace part of the model twice."""
    import inspect

    try:
        return name in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def walk_model(
    model: nn.Module,
    example_inputs: Mapping[str, Any] | tuple | torch.Tensor | None = None,
    *,
    claim_rules: Sequence[ClaimRule] = (),
    execution: str = "fake",
    strict: bool = True,
    seq_len: int = 8,
) -> WalkResult:
    """Walk a model: enumerate its named tensors, trace one forward, resolve
    every matmul-family operand, and claim every node.

    Args:
        model: any ``nn.Module``. For ``execution="fake"``, meta-loaded is
            the intended (and cheapest) state.
        example_inputs: kwargs mapping, positional tuple, or a single tensor
            for the traced forward. Defaults to ``input_ids`` of shape
            ``(1, seq_len)`` on the model's device, with ``use_cache=False``
            when the forward accepts it.
        claim_rules: ordered :class:`ClaimRule` list; first match claims a
            node. Model profiles supply these via ``walk_claim_rules()``.
        execution: ``"fake"`` runs the forward under ``FakeTensorMode``
            (no weight I/O, works on meta); ``"real"`` runs a plain forward
            (root-B fallback for models fake tensors cannot execute).
        strict: raise :class:`WalkError` on any failure. ``strict=False``
            returns the result with ``failures`` populated instead.
        seq_len: length of the synthesized default input.

    Returns:
        A :class:`WalkResult` — the single enumeration downstream consumers
        derive from.

    Raises:
        WalkError: when ``strict`` and a matmul-fed node is unclaimed or a
            floating multiplicand cannot be resolved to any named tensor.
    """
    if execution not in ("fake", "real"):
        raise ValueError(f"execution must be 'fake' or 'real', got {execution!r}")

    nodes, named_storages = _named_tensor_index(model)
    device = _model_device(model)
    if execution == "real" and device.type == "meta":
        raise ValueError(
            "execution='real' needs materialized weights; this model is on "
            "the meta device. Use execution='fake', or load real weights.")
    default_inputs = example_inputs is None
    if default_inputs:
        example_inputs = _default_example_inputs(model, seq_len, device)
    # Trace-time provenance: captured before the forward, because none of it
    # is reconstructible afterward (see WalkProvenance).
    provenance = capture_walk_provenance(
        model,
        execution=execution,
        example_inputs=example_inputs,
        seq_len=seq_len if default_inputs else None,
        claim_rules=claim_rules,
        used_default_inputs=default_inputs,
    )

    interceptor = WeightUseInterceptor(named_storages)
    input_tensors = list(_iter_tensors(
        list(example_inputs.values())
        if isinstance(example_inputs, Mapping) else example_inputs))
    interceptor.mark_inputs(input_tensors)

    executed: set[str] = set()
    hook_handles = []
    module_names = dict(model.named_modules())

    def _pre_hook(qname):
        def hook(module, args, kwargs=None):
            executed.add(qname)
            interceptor.module_stack.append(qname)
        return hook

    def _post_hook(qname):
        def hook(module, args, output):
            if interceptor.module_stack and \
                    interceptor.module_stack[-1] == qname:
                interceptor.module_stack.pop()
        return hook

    for qname, module in module_names.items():
        hook_handles.append(
            module.register_forward_pre_hook(_pre_hook(qname)))
        hook_handles.append(
            module.register_forward_hook(_post_hook(qname)))

    try:
        with torch.no_grad():
            if execution == "fake":
                from torch._subclasses.fake_tensor import FakeTensorMode

                fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
                with interceptor, fake_mode:
                    _call_forward(model, example_inputs)
            else:
                with interceptor:
                    _call_forward(model, example_inputs)
    finally:
        for handle in hook_handles:
            handle.remove()

    result = _assemble(
        nodes, interceptor, executed, module_names, claim_rules, execution,
        provenance=provenance)
    interceptor.release()
    if strict:
        result.raise_if_failed()
    return result


def _assemble(
    nodes: list[WalkNode],
    interceptor: WeightUseInterceptor,
    executed: set[str],
    module_names: dict[str, nn.Module],
    claim_rules: Sequence[ClaimRule],
    execution: str,
    provenance: WalkProvenance | None = None,
) -> WalkResult:
    node_by_name = {n.name: n for n in nodes}

    # Resolve operands -> edges / unresolved.
    edge_counts: dict[tuple, int] = {}
    unresolved: list[UnresolvedOperand] = []
    for record in interceptor.records:
        all_shapes = tuple(shape for _, shape, _, _, _ in record.operands)
        for index, (key, shape, dtype, role, floating) in enumerate(
                record.operands):
            entry = interceptor.origin.get(key) if key is not None else None
            if entry is not None:
                names, via = entry
                primary = names[0]
                node = node_by_name.get(primary)
                edge_key = (
                    primary, tuple(names[1:]), record.op, record.equation,
                    role, index, shape, dtype, all_shapes,
                    node.stored_bytes if node else 0, record.module, via,
                )
                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
                continue
            if key is not None and key in interceptor.computed:
                continue  # an activation / traced intermediate
            unresolved.append(UnresolvedOperand(
                op=record.op, equation=record.equation, module=record.module,
                operand_index=index, operand_shape=shape,
                operand_dtype=dtype, role=role, is_floating=floating,
            ))

    edges = tuple(sorted(
        (WalkEdge(
            param=k[0], param_aliases=k[1], op=k[2], equation=k[3],
            role=k[4], operand_index=k[5], operand_shape=k[6],
            operand_dtype=k[7], operand_shapes=k[8], stored_bytes=k[9],
            module=k[10], via=k[11], calls=count)
         for k, count in edge_counts.items()),
        key=lambda e: (e.param, e.module, e.op, e.operand_index, e.role,
                       e.operand_shape, e.via, e.equation or ""),
    ))

    embedding_uses = tuple(sorted(
        {EmbeddingUse(
            param=names[0], param_aliases=tuple(names[1:]), module=module)
         for names, module in interceptor.embedding_records
         if names is not None},
        key=lambda u: (u.param, u.module),
    ))

    # Claims.
    claims = apply_claim_rules(nodes, claim_rules)
    unclaimed = tuple(sorted(
        n.name for n in nodes if n.name not in claims))

    # Failures.
    failures: list[WalkFailure] = []
    seen_failure_nodes: set[tuple[str, str]] = set()
    for edge in edges:
        if edge.role != "multiplicand":
            continue
        for name in (edge.param, *edge.param_aliases):
            if name in claims or (name, edge.op) in seen_failure_nodes:
                continue
            seen_failure_nodes.add((name, edge.op))
            failures.append(WalkFailure(
                kind="unclaimed", node=name, op=edge.op,
                equation=edge.equation, module=edge.module,
                detail=(
                    f"parameter {name!r} (shape "
                    f"{list(node_by_name[name].shape)}, "
                    f"{node_by_name[name].stored_bytes} bytes) feeds this op "
                    "and no claim rule matched it"),
            ))
    for use in embedding_uses:
        for name in (use.param, *use.param_aliases):
            if name in claims or (name, "embedding") in seen_failure_nodes:
                continue
            seen_failure_nodes.add((name, "embedding"))
            failures.append(WalkFailure(
                kind="unclaimed", node=name, op="embedding", equation=None,
                module=use.module,
                detail=(f"embedding weight {name!r} is consumed by "
                        "F.embedding and no claim rule matched it"),
            ))
    for operand in unresolved:
        if not (operand.is_floating and operand.role == "multiplicand"):
            continue
        failures.append(WalkFailure(
            kind="unresolved", node=None, op=operand.op,
            equation=operand.equation, module=operand.module,
            detail=(
                f"floating operand #{operand.operand_index} (shape "
                f"{list(operand.operand_shape)}, {operand.operand_dtype}) "
                "matches no named parameter or buffer and was not computed "
                "by the traced forward — a weight this walk cannot name "
                "(was it .to()'d or reconstructed outside the forward?)"),
        ))

    # Trace coverage.
    containers = tuple(sorted(
        qname for qname, module in module_names.items()
        if isinstance(module, _CONTAINER_CLASSES)))
    container_set = set(containers)
    executed_names = tuple(sorted(
        q for q in executed if q not in container_set))
    not_executed = tuple(sorted(
        qname for qname in module_names
        if qname not in executed and qname not in container_set))

    return WalkResult(
        nodes=tuple(sorted(nodes, key=lambda n: (n.name, n.kind))),
        edges=edges,
        claims=claims,
        unclaimed=unclaimed,
        embedding_uses=embedding_uses,
        unresolved_operands=tuple(unresolved),
        failures=tuple(failures),
        trace_coverage=TraceCoverage(
            executed=executed_names,
            not_executed=not_executed,
            containers=containers,
        ),
        execution=execution,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# The export gate
#
# A STRUCTURED verdict over a WalkResult. The refusal branches on fields —
# refusal_kinds, failure kind, node/op/equation/module lists — never on the
# prose `detail` strings, which exist for the human reading the log.
#
# Tensor-Parallel stance (invariants, 2026-08-22): the decision unit is the
# whole logical tensor, so node identity and dispositions are TP-degree
# invariant by construction; `stored_bytes` everywhere here is TOTAL bytes of
# the logical tensor with per-device accounting deferred to an additive
# `shard_policy` annotation; and KNOWN_FAILURE_KINDS is a closed set whose
# unknown members refuse, so a future TP category (e.g.
# "tp_group_boundary_misaligned" — quantization group boundary vs shard
# boundary at degree N) makes even an unupgraded gate fail closed.
# ---------------------------------------------------------------------------

WALK_GATE_SCHEMA = "prismaquant.model_walk_gate.v1"

#: Per-run escape hatch for TRACE INCOMPLETENESS ONLY. Same doctrine as the CB
#: route-status override: the value must be the REASON, and it is stamped into
#: the verdict's provenance. It never excuses an unclaimed node or an
#: unresolved multiplicand — claims have a first-class mechanism
#: (`pin(reason)` / `exclude(reason)` in ModelProfile.walk_claim_rules()) and
#: a side channel around it would be the silent-green this gate exists to kill.
WALK_GATE_OVERRIDE_ENV = "PRISMAQUANT_WALK_GATE_OVERRIDE"

TRACE_COMPLETE = "complete"
TRACE_INCOMPLETE = "incomplete"

#: The closed vocabulary of WalkFailure.kind this gate understands. Anything
#: else refuses (fail-closed catch-all for future categories).
KNOWN_FAILURE_KINDS = frozenset({"unclaimed", "unresolved"})

_KIND_UNCLAIMED = "unclaimed_node"
_KIND_UNRESOLVED = "unresolved_floating_multiplicand"
_KIND_UNKNOWN_FAILURE = "unknown_walk_failure_kind"
_KIND_TRACE_INCOMPLETE = "incomplete_trace"
_KIND_DECIDED_UNPRICED = "decided_but_unpriced_node"


class WalkGateRefusal(RuntimeError):
    """Export refused by the discovery-walker coverage gate."""


@dataclasses.dataclass(frozen=True)
class WalkGateVerdict:
    """The gate's result: structured provenance plus what it decided.

    ``refusal_reason`` is prose for humans; every consumer (the CLI exit
    code, run-pipeline.sh, a future shipcard stamp) branches only on
    :attr:`refused` and the structured fields of :attr:`provenance`.
    """

    provenance: dict
    refused: bool
    refusal_kinds: tuple[str, ...]
    refusal_reason: str = ""


def evaluate_walk_gate(
    result: WalkResult | None,
    *,
    trace_status: str = TRACE_COMPLETE,
    trace_error_class: str = "",
    override_reason: str | None = None,
    unpriced_decides: Sequence[Mapping] = (),
    scope: Mapping | None = None,
) -> WalkGateVerdict:
    """Project one walk onto the fail-closed export verdict.

    Args:
        result: the walk output; ``None`` means the traced forward itself
            aborted (``trace_status`` is then forced to
            ``TRACE_INCOMPLETE``) and no coverage claim is possible.
        trace_status: ``"complete"`` or ``"incomplete"``. An incomplete trace
            discovers only what executed before the abort, so gating on it
            would under-discover exactly like the pipeline enumerations this
            walker replaces: it refuses unless ``override_reason`` is given.
        trace_error_class: exception class name when the trace aborted
            (recorded, never branched on).
        override_reason: explicit per-run reason excusing trace incompleteness
            only. Defaults to :data:`WALK_GATE_OVERRIDE_ENV`. Claim failures
            refuse regardless of any override.
        unpriced_decides: structured entries for the contradiction class —
            a node claimed ``decide`` that no pricing consumer can actually
            reach (probe-skip class, probe-exclude regex). The gemma4 router
            shipped exactly this way: decided by rule 9, excluded from probe
            inventory by name. Each entry carries node/module_class/reason_code;
            presence refuses, with no override.
        scope: structured declaration of what this gate run asserts over
            (profile name, rules source, rule-family census) — enabling is
            provable per profile from the report itself, never prose.

    Returns:
        A :class:`WalkGateVerdict` whose ``provenance`` is JSON-serializable
        and self-describing (decision unit, byte-accounting convention).
    """
    if result is None:
        trace_status = TRACE_INCOMPLETE

    unclaimed_nodes: list[dict] = []
    unresolved_operands: list[dict] = []
    kinds_seen: set[str] = set()
    if result is not None:
        kinds_seen.update(f.kind for f in result.failures)
        seen: set[tuple] = set()
        for f in result.failures:
            if f.kind == "unclaimed":
                entry = {
                    "node": f.node,
                    "op": f.op,
                    "equation": f.equation,
                    "module": f.module,
                }
                key = tuple(sorted(entry.items()))
                if key not in seen:
                    seen.add(key)
                    unclaimed_nodes.append(entry)
        unresolved_operands = [
            {
                "op": u.op,
                "equation": u.equation,
                "module": u.module,
                "operand_index": u.operand_index,
                "operand_shape": list(u.operand_shape),
                "operand_dtype": u.operand_dtype,
            }
            for u in result.unresolved_operands
            if u.is_floating and u.role == "multiplicand"
        ]

    disposition_counts = {"decide": 0, "pin": 0, "exclude": 0}
    if result is not None:
        for claim in result.claims.values():
            disposition_counts[claim.disposition] += 1

    fed_unclaimed = len(unclaimed_nodes)
    unfed_unclaimed = (
        len(result.unclaimed) - fed_unclaimed if result is not None else 0
    )

    base: dict = {
        "schema": WALK_GATE_SCHEMA,
        "policy": "fail_closed",
        # TP invariant 1: identity is the logical tensor; sharding is a
        # load-time concern and never renames or re-claims a node.
        "decision_unit": "whole_logical_tensor",
        # TP invariant 2: say which byte convention the schema carries.
        "byte_accounting": {
            "convention": "total_logical_tensor_bytes",
            "shard_policy": None,  # reserved; additive when TP lands
        },
        "trace_status": trace_status,
        "trace_error_class": trace_error_class,
        "nodes_total": len(result.nodes) if result is not None else None,
        "edges_total": len(result.edges) if result is not None else None,
        "claims_by_disposition": disposition_counts,
        "matmul_fed_unclaimed_total": fed_unclaimed,
        "unclaimed_matmul_fed_nodes": sorted(
            unclaimed_nodes,
            key=lambda e: (e["node"] or "", e["op"], e["equation"] or ""),
        ),
        # Visible, non-fatal: nodes nothing consumed in the trace (a bias, a
        # dormant module). Reported so silence can never masquerade as
        # coverage, but they are not refusals.
        "unclaimed_unfed_total": max(unfed_unclaimed, 0),
        "unresolved_floating_multiplicands": unresolved_operands,
        "failure_kinds_seen": sorted(kinds_seen),
        "decided_but_unpriced_nodes": [dict(e) for e in unpriced_decides],
        "scope": dict(scope or {}),
        "override": _gate_override_record(override_reason),
    }

    kinds: list[str] = []
    unknown = sorted(kinds_seen - KNOWN_FAILURE_KINDS)
    if unknown:
        kinds.append(_KIND_UNKNOWN_FAILURE)
    if unclaimed_nodes:
        kinds.append(_KIND_UNCLAIMED)
    if unresolved_operands:
        kinds.append(_KIND_UNRESOLVED)
    if unpriced_decides:
        kinds.append(_KIND_DECIDED_UNPRICED)
    if trace_status != TRACE_COMPLETE:
        kinds.append(_KIND_TRACE_INCOMPLETE)

    base["refused"] = bool(kinds)
    base["refusal_kinds"] = list(kinds)

    if not kinds:
        return WalkGateVerdict(
            provenance=base, refused=False, refusal_kinds=())

    claim_refusals = [
        k for k in kinds
        if k in (_KIND_UNKNOWN_FAILURE, _KIND_UNCLAIMED, _KIND_UNRESOLVED,
                 _KIND_DECIDED_UNPRICED)
    ]
    if claim_refusals:
        # No override reaches here, ever: claims are pinned/excluded/decided
        # with reasons in the profile rules, not waived at export time.
        refused = True
    elif override_reason:
        base["override_excused_trace_only"] = True
        refused = False
    else:
        refused = True

    return WalkGateVerdict(
        provenance=base,
        refused=refused,
        refusal_kinds=tuple(kinds),
        refusal_reason=_refusal_text(base, kinds, unknown, bool(override_reason)),
    )


def find_decided_but_unpriced(
    result: WalkResult,
    model: nn.Module,
    profile,
) -> tuple[dict, ...]:
    """Surface the contradiction the gemma4 router shipped under: a node
    claimed ``decide`` that no pricing consumer can reach.

    A ``decide`` claim is a promise — "the allocator prices this tensor".
    When the probe's own enumeration cannot see the node (owner class in
    ``probe_skip_module_class_names``, or the node's name matches the
    baseline/profile Linear-exclude regexes), and it is not priced through
    the packed-expert path either, that promise is false and this returns a
    structured entry for it. The gate refuses on any entry; the fix is to
    correct the CLAIM (usually to a router-style pin), never to widen the
    probe.

    Imports prismaquant lazily so the walker core stays torch-only.
    """
    from prismaquant.incremental_probe import _BASE_LINEAR_EXCLUDE
    from prismaquant.sensitivity_probe import _is_packed_experts_module

    try:
        extra = str(profile.probe_linear_exclude_extra() or "")
    except AttributeError:
        extra = ""
    excludes = (_BASE_LINEAR_EXCLUDE,) if not extra else (
        _BASE_LINEAR_EXCLUDE, extra)

    entries: list[dict] = []
    for node in result.nodes:
        claim = result.claims.get(node.name)
        if claim is None or claim.disposition != "decide":
            continue
        try:
            module = model.get_submodule(node.owner_module)
        except (AttributeError, KeyError):
            entries.append({
                "node": node.name,
                "owner_module": node.owner_module,
                "module_class": node.module_class,
                "reason_code": "owner_module_missing",
            })
            continue
        if profile.should_probe_linear(node.owner_module, module):
            # The hook gate accepts the module; the NAME regexes are the
            # remaining way pricing can silently skip it.
            if not any(re.search(rx, node.name) for rx in excludes):
                continue
            reason_code = "probe_linear_excluded"
        elif _is_packed_experts_module(module, profile):
            # Priced through install_packed_expert_hooks instead.
            continue
        else:
            reason_code = "owner_not_probe_priceable"
        entries.append({
            "node": node.name,
            "owner_module": node.owner_module,
            "module_class": node.module_class,
            "reason_code": reason_code,
        })
    return tuple(sorted(entries, key=lambda e: e["node"]))


def require_walk_coverage(
    result: WalkResult | None,
    **kwargs,
) -> dict:
    """Run the gate and raise :class:`WalkGateRefusal` on a refusal.

    Returns the provenance payload for stamping next to the artifact's other
    gate records.
    """
    verdict = evaluate_walk_gate(result, **kwargs)
    if verdict.refused:
        raise WalkGateRefusal(verdict.refusal_reason)
    return verdict.provenance


def _gate_override_record(reason: str | None) -> dict | None:
    if not reason:
        return None
    return {"env": WALK_GATE_OVERRIDE_ENV, "reason": reason}


def _refusal_text(base: dict, kinds: Sequence[str],
                  unknown: Sequence[str], overridden: bool) -> str:
    lines = [f"model-walk export gate refused [{', '.join(kinds)}]:"]
    for entry in base["unclaimed_matmul_fed_nodes"]:
        where = entry["op"] + (
            f" '{entry['equation']}'" if entry["equation"] else "")
        lines.append(
            f"  [unclaimed] {entry['node']} fed to {where} "
            f"in module '{entry['module'] or '<root>'}'"
        )
    for op in base["unresolved_floating_multiplicands"]:
        where = op["op"] + (f" '{op['equation']}'" if op["equation"] else "")
        lines.append(
            f"  [unresolved] floating operand #{op['operand_index']} "
            f"(shape {op['operand_shape']}) fed to {where} "
            f"in module '{op['module'] or '<root>'}'"
        )
    for entry in base["decided_but_unpriced_nodes"]:
        lines.append(
            f"  [decided-but-unpriced] {entry.get('node')} "
            f"(class {entry.get('module_class') or '?'}, "
            f"reason_code {entry.get('reason_code')}) is claimed decide but "
            "no pricing consumer can reach it"
        )
    if _KIND_UNKNOWN_FAILURE in kinds:
        lines.append(
            f"  [unknown-kind] walk emitted failure kinds outside the known "
            f"vocabulary {sorted(KNOWN_FAILURE_KINDS)}: {list(unknown)} — "
            "refusing without interpreting them"
        )
    if _KIND_TRACE_INCOMPLETE in kinds:
        lines.append(
            "  [incomplete-trace] the traced forward aborted, so coverage "
            f"was evaluated over a partial discovery (last error class: "
            f"{base['trace_error_class'] or 'n/a'})"
        )
    if overridden and not any(
        k in (_KIND_UNCLAIMED, _KIND_UNRESOLVED, _KIND_UNKNOWN_FAILURE,
              _KIND_DECIDED_UNPRICED)
        for k in kinds
    ):
        lines.append(
            f"  override {WALK_GATE_OVERRIDE_ENV} excuses trace "
            "incompleteness ONLY; it is stamped into the report"
        )
    lines.append(
        "Fixes, in order of preference: pin/exclude/decide the named node "
        "with a reasoned ClaimRule in ModelProfile.walk_claim_rules(); make "
        "the fake trace executable; or --execution real with materialized "
        f"weights. Trace-incompleteness alone may ship via "
        f"{WALK_GATE_OVERRIDE_ENV}=<reason> (stamped). Claim failures have "
        "no override."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entrypoint (intake + export gate)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Walk a checkpoint directory and apply the fail-closed export gate.

    Exit codes follow the repo's guard convention: 0 passed, 2 refused.
    """
    import argparse
    import json
    import os
    import pathlib
    import sys

    ap = argparse.ArgumentParser(
        prog="python3 -m prismaquant.model_walk",
        description=(
            "Discovery walker (R5): enumerate every named tensor, trace one "
            "forward, resolve matmul operands by storage identity, claim "
            "every node, and refuse on any unclaimed matmul-fed parameter "
            "or unresolved floating multiplicand."
        ),
    )
    ap.add_argument("--model", required=True,
                    help="Source HF checkpoint directory.")
    ap.add_argument("--output", default=None,
                    help="Write the walk + structured gate verdict JSON here.")
    ap.add_argument("--seq-len", type=int, default=8,
                    help="Length of the synthesized default input_ids.")
    ap.add_argument("--execution", choices=("fake", "real"), default="fake",
                    help="'fake' traces under FakeTensorMode (meta model, no "
                         "weight I/O); 'real' runs a plain forward and "
                         "requires --materialize.")
    ap.add_argument("--materialize", action="store_true",
                    help="Load real weights instead of meta-loading (needed "
                         "for --execution real; sized like the checkpoint).")
    ap.add_argument("--dtype", default="bf16",
                    help="dtype for --materialize loads (default bf16).")
    ap.add_argument("--rules", choices=("profile", "none"), default="profile",
                    help="'profile' applies detect_profile().walk_claim_rules(); "
                         "'none' applies no rules (every matmul-fed node "
                         "refuses — useful as a self-test of the gate).")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--override-reason", default=None,
                    help="Explicit reason excusing TRACE INCOMPLETENESS only "
                         f"(same record as {WALK_GATE_OVERRIDE_ENV}). Never "
                         "valid against claim failures.")
    args = ap.parse_args(argv)

    from prismaquant.model_profiles.registry import (
        detect_profile_with_warning,
    )

    profile = detect_profile_with_warning(args.model, entrypoint="model_walk")

    import torch
    from transformers import AutoConfig, AutoModel

    try:
        from transformers import AutoModelForCausalLM
        have_causal_lm = True
    except Exception:  # pragma: no cover - transformers always ships it
        have_causal_lm = False

    cfg = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code)
    load_kwargs: dict = {"trust_remote_code": args.trust_remote_code}
    if args.materialize:
        load_kwargs["dtype"] = getattr(torch, args.dtype)
        load_kwargs["low_cpu_mem_usage"] = True
        model_class_used = ""
        if have_causal_lm:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model, **load_kwargs)
                model_class_used = "causal_lm"
            except Exception:
                model = None
        if not have_causal_lm or model is None:
            model = AutoModel.from_pretrained(args.model, **load_kwargs)
            model_class_used = model_class_used or "base"
    else:
        device_ctx = torch.device("meta")
        model = None
        model_class_used = ""
        if have_causal_lm:
            try:
                with device_ctx:
                    model = AutoModelForCausalLM.from_config(cfg, **load_kwargs)
                model_class_used = "causal_lm"
            except Exception:
                model = None
        if model is None:
            with device_ctx:
                model = AutoModel.from_config(cfg, **load_kwargs)
            model_class_used = model_class_used or "base"
    model.eval()

    rules = profile.walk_claim_rules() if args.rules == "profile" else ()
    print(f"[model-walk] profile={getattr(profile, 'name', type(profile).__name__)}"
          f" model_class={model_class_used} rules={len(rules)}"
          f" execution={args.execution}")

    trace_status = TRACE_COMPLETE
    trace_error_class = ""
    result = None
    try:
        result = walk_model(
            model,
            claim_rules=rules,
            execution=args.execution,
            strict=False,
            seq_len=args.seq_len,
        )
    except Exception as exc:  # noqa: BLE001 - the abort IS the finding
        trace_status = TRACE_INCOMPLETE
        trace_error_class = type(exc).__name__
        print(f"[model-walk] trace aborted: {type(exc).__name__}: {exc}",
              flush=True)

    override_reason = args.override_reason or os.environ.get(
        WALK_GATE_OVERRIDE_ENV) or None

    # The gemma4-router contradiction class: a decide claim no pricing
    # consumer can reach. Computed against the live profile + module tree,
    # fed to the gate as structured entries.
    unpriced_decides: tuple[dict, ...] = ()
    scope: dict = {}
    if result is not None:
        try:
            unpriced_decides = find_decided_but_unpriced(
                result, model, profile)
        except Exception as exc:  # noqa: BLE001 - checker failure is loud
            print(f"[model-walk] WARNING: decided-but-unpriced check failed "
                  f"({type(exc).__name__}: {exc}); treating as refusal input",
                  file=sys.stderr)
            unpriced_decides = ({
                "node": "<checker>",
                "module_class": "",
                "reason_code": f"check_failed:{type(exc).__name__}",
            },)
        rules_json = result.provenance.claim_rules if \
            result.provenance is not None else ()
        scope = {
            "profile": getattr(profile, "name", type(profile).__name__),
            "rules_source": args.rules,
            "claim_rule_count": len(rules),
            "router_pin_rules": sum(
                1 for r in rules_json
                if r.get("disposition") == "pin"
                and "router" in json.dumps(r).lower()),
            "packed_expert_decide_rules": sum(
                1 for r in rules_json
                if r.get("disposition") == "decide"
                and "expert" in json.dumps(r).lower()),
        }

    verdict = evaluate_walk_gate(
        result,
        trace_status=trace_status,
        trace_error_class=trace_error_class,
        override_reason=override_reason,
        unpriced_decides=unpriced_decides,
        scope=scope,
    )

    report = {
        "schema": WALK_GATE_SCHEMA,
        "walk_artifact_schema": SCHEMA,
        "context": {
            "model_path": args.model,
            "profile": getattr(profile, "name", type(profile).__name__),
            "auto_model_class": model_class_used,
            "execution": args.execution,
            "materialized": bool(args.materialize),
            "rules_source": args.rules,
        },
        "gate": verdict.provenance,
        "provenance": (
            result.provenance.to_json_dict()
            if result is not None and result.provenance is not None
            else None
        ),
        "walk": result.to_json_dict() if result is not None else None,
    }
    if args.output:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(f"[model-walk] wrote {out}")

    counts = verdict.provenance["claims_by_disposition"]
    print(
        "[model-walk] nodes="
        f"{verdict.provenance['nodes_total']} "
        f"claims={counts} "
        f"unclaimed_matmul_fed={verdict.provenance['matmul_fed_unclaimed_total']} "
        f"unresolved={len(verdict.provenance['unresolved_floating_multiplicands'])} "
        f"trace={verdict.provenance['trace_status']}"
    )
    if verdict.refused:
        print(verdict.refusal_reason, file=sys.stderr)
        print("[model-walk] GATE: REFUSED", file=sys.stderr)
        return 2
    print("[model-walk] GATE: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
