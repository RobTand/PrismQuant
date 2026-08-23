"""Expected per-token decode READ bytes for a quantization assignment.

Why this exists
---------------
The allocator minimizes KL under a **disk-byte** budget.  Decode throughput
is not governed by disk bytes: it is governed by the bytes the serving
runtime must actually stream *per generated token*.  On a sparse MoE those
two objectives diverge violently, because a dense weight is read on every
token while a routed expert stack is read only when the router selects it.

Measured 2026-08-21 on the shipped DSv4-Flash 87 GB artifact: the dense path
is **8.3% of the checkpoint but 76.8% of decode read traffic**, and the whole
artifact costs **8.0576 GB read per token** at batch 1.  A byte budget cannot
see that, so it systematically overspends decode bandwidth on the dense path.
This module is the *measurement* half of closing that gap (principle 1: it is
a measurement gap, not an optimizer gap).  It deliberately changes no
allocation; pricing read bytes inside the DP is a separate decision.

The definition
--------------
::

    read_bytes_per_token = Σ_tensor stored_bytes(tensor) × read_probability(tensor)

with exactly one read probability per tensor class (see
:data:`READ_CLASS_TABLE`, which is the single authority for that mapping):

===========================  =====================  ==========================
class                        read probability       what lands here
===========================  =====================  ==========================
``routed_experts``           ``topk / E``           routed MoE expert stacks
``dense``                    ``1.0``                allocator-assigned units
                                                    that are always active
``held_fixed``               ``1.0``                always-active tensors the
                                                    allocator never decided —
                                                    norms, biases, routers,
                                                    a pinned ``lm_head``.
                                                    (DSv4's grouped ``wo_a``
                                                    used to be named here as
                                                    "probe-skipped"; since the
                                                    grouped Fisher accumulator
                                                    landed it is priced and
                                                    moves to ``dense`` exactly
                                                    when an assignment covers
                                                    it — same p=1.0 either
                                                    way.)
``excluded_embedding``       ``0.0`` (excluded)     the input embedding table
                                                    of an UNTIED model: one row
                                                    is gathered per token, not
                                                    the table.  A *tied* table
                                                    is the output projection
                                                    and lands in ``held_fixed``
                                                    — see below
``excluded_indexed_lookup``  ``0.0`` (excluded)     integer tables addressed by
                                                    token id (DSv4's
                                                    ``ffn.gate.tid2eid``): one
                                                    row per token, like the
                                                    embedding
``excluded_mtp``             ``0.0`` (excluded)     MTP / draft sidecar — read
                                                    only when spec-decode is
                                                    on; see the honest-default
                                                    note below
``excluded_non_text_graph``  ``0.0`` (excluded)     tensors the model profile
                                                    itself declines to map
                                                    into the live text graph
                                                    (vision / audio towers)
``resident_codebooks``       n/a — reported as      CB codebook tables: tiny,
                             ``resident_bytes``     cache-resident, and not
                                                    per-token stream traffic
===========================  =====================  ==========================

``topk / E`` is exact as an *expectation* under the per-layer-uniform serving
invariant PrismaQuant already enforces (experts are uniform per layer, mixed
across layers): every expert in a layer's stack stores the same bytes, so
routing skew redistributes which experts are read without changing the
expected bytes read.  Where a lane breaks that invariant — a CB split-stack
whose sub-stacks carry different bytes per expert — the number becomes an
expectation under *uniform* routing and says so in ``routing.exactness``.

The embedding is excluded because it is *gathered*, not streamed — but that
is only true while it is nothing else.  Under ``tie_word_embeddings`` the same
table **is** the output projection, and the logits matmul streams all of it on
every token.  Which case an artifact is in is decided by observation, not by
name: the logits projection is streamed exactly once per token, so the
embedding is excluded when the checkpoint carries a separate ``lm_head``
tensor and is ``held_fixed`` when it does not; ``resolve_embedding_disposition``
is where that is decided.  The config's ``tie_word_embeddings`` is the
cross-check, and a
config that declares *untied* on a checkpoint with no output projection raises
rather than picking one of the two stories.  Excluding a tied table would drop
one of the largest always-active tensors in the model (Qwen3-0.6B ties; so
does LFM2.5).

Indexed lookup tables take **three** facts, not one
---------------------------------------------------
An integer tensor is not automatically an index table, and each weaker rule was
falsified on a real artifact before this one was settled:

* *dtype only* — the packed weight payload of both quantized lanes is an
  integer dtype and is streamed in full (``U8`` is 81.65 GB of ``cb_qweight``
  on the shipped DSv4 CB body and 15.7 GB of NVFP4 ``weight_packed`` on
  Ornith-1.5-35B).  Classifying by dtype alone excludes 94% of that artifact.
* *dtype + vocabulary-keyed leading axis* — a **quantized ``lm_head``** is
  both, and it is streamed in full.  On the `embed-smoke` CB-head export that
  rule dropped 857,736 B of real logits traffic: an **under**-count, the
  dangerous direction.

So ``excluded_indexed_lookup`` requires all three: *integer dtype* (it cannot
be a float matmul operand on its own) **AND** ``shape[0] == vocab_size`` (a
decode step addresses one row by token id) **AND** *the module is not declared
a quantized weight* — neither by a float scale sidecar (nothing turns those
integers into numbers) nor by the artifact's own
``config_groups[*].targets``.  Both halves of the third fact are needed: a CB
payload keeps its scales in the codebook sidecar, so ``lm_head.cb_qweight`` has
no scale tensor in the shard set at all, and only the declared ``targets:
["lm_head"]`` covers it.  Everything is read from the safetensors headers and
the artifact's own config; the scale suffixes come from
``footprint._SIDECAR_SUFFIXES``, the constant the byte partition already folds
on.  DSv4's ``ffn.gate.tid2eid`` — I64 ``[129280, 6]``, token id to its six
expert ids, no scale on ``ffn.gate`` and no target naming it — is the case that
survives all three.

The residual direction is deliberate: anything the three facts cannot establish
stays in ``held_fixed`` at ``p=1``, which **over**-counts read traffic.  For a
bandwidth figure that is the safe direction, and it is the opposite of the
failure this module exists to prevent.

Honest defaults for the genuinely ambiguous classes
---------------------------------------------------
The MTP sidecar is read every token when the artifact is served with
spec-decode and never when it is not, and nothing in the recipe says which.
It is **excluded but itemized** (``excluded.mtp_bytes``) so a caller serving
with spec-decode can add it back exactly.  The same applies to vision/audio
towers, which a text decode never touches.  Nothing is silently dropped: the
ledger reconciles against ``footprint.assignment_artifact_bytes`` to the byte
before any probability is applied, and refuses if it does not.

Scope
-----
Weights only, batch 1, greedy decode.  KV-cache traffic, activations, and
safetensors container metadata are outside it — so a figure from here is a
**lower bound** on real decode traffic, which is the honest direction for a
bandwidth ceiling.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import format_registry as fr
from . import footprint as fp
from .allocator_solver import _shape_from_stats
from .cb_export_config import CODEBOOK_TENSOR_PREFIX
from .name_projection import (
    DECLARED_OUT_OF_GRAPH,
    MAPPED,
    NameProjection,
    strip_weight_leaf,
)
from .nvfp4_cb_footprint import is_cb_format

SCHEMA = "prismaquant.read_traffic.v1"

#: Human-readable scope of every number this module produces.  It travels
#: with the value wherever it is stamped, because a published bandwidth
#: figure without its convention is not checkable (principle 12).
READ_SCOPE = (
    "weights only; expected bytes streamed per generated token at batch 1; "
    "KV cache, activations, and safetensors container metadata excluded; "
    "routed-expert stacks weighted by num_experts_per_tok / n_routed_experts"
)

#: THE mapping from tensor class to read probability.  Every consumer reads
#: it from here; there is no second copy.  ``None`` means the class carries
#: no per-token stream traffic at all and is reported separately.
READ_CLASS_TABLE: dict[str, float | None] = {
    "routed_experts": None,  # resolved at run time to topk / E
    "dense": 1.0,
    "held_fixed": 1.0,
    "excluded_embedding": 0.0,
    "excluded_indexed_lookup": 0.0,
    "excluded_mtp": 0.0,
    "excluded_non_text_graph": 0.0,
    "resident_codebooks": None,  # resident, never streamed per token
}

#: Classes whose bytes sum into ``read_bytes_per_token``.
STREAMED_CLASSES = ("dense", "routed_experts", "held_fixed")
#: Classes reported under ``excluded`` — real bytes, zero per-token traffic.
EXCLUDED_CLASSES = (
    "excluded_embedding", "excluded_indexed_lookup", "excluded_mtp",
    "excluded_non_text_graph")

#: safetensors dtype names that cannot be a float matmul operand *on their
#: own*.  Necessary but NOT sufficient for the lookup class -- both quantized
#: lanes store the packed weight payload as ``U8``, and it is read in full.
INTEGER_SAFETENSORS_DTYPES = frozenset({
    "BOOL", "U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64"})

#: HF config keys that declare the vocabulary size, most specific first.
_VOCAB_SIZE_KEYS = ("vocab_size", "padded_vocab_size", "n_vocab")

#: HF config keys that declare the routed-expert count, most specific first.
_EXPERT_COUNT_KEYS = (
    "n_routed_experts", "num_local_experts", "num_experts", "n_experts")
#: HF config keys that declare how many routed experts a token activates.
_EXPERTS_PER_TOK_KEYS = (
    "num_experts_per_tok", "num_experts_per_token", "moe_top_k",
    "num_active_experts")


class ReadTrafficError(ValueError):
    """A tensor could not be classified, priced, or reconciled.

    Always raised rather than defaulted around: a silent zero in a
    per-tensor score ranks the broken arm first (project memory,
    ``silent_zero_scores_rank_broken_arms_first``), and the same is true of
    a silently-omitted tensor in a bandwidth ledger.
    """


@dataclass(frozen=True)
class RoutingFactor:
    """The routed-expert read probability and where every term came from."""

    num_experts_per_tok: int
    n_routed_experts: int
    source: str
    exactness: str

    @property
    def read_probability(self) -> float:
        return float(self.num_experts_per_tok) / float(self.n_routed_experts)


# ---------------------------------------------------------------------------
# Routing factor
# ---------------------------------------------------------------------------

def _config_scopes(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """The config dicts a routing declaration may legitimately live in."""
    scopes: list[Mapping[str, Any]] = [config]
    for key in ("text_config", "language_model_config"):
        inner = config.get(key)
        if isinstance(inner, Mapping):
            scopes.append(inner)
    return tuple(scopes)


def _first_positive_int(
    scopes: Iterable[Mapping[str, Any]], keys: Iterable[str],
) -> tuple[int, str] | None:
    for scope_idx, scope in enumerate(scopes):
        for key in keys:
            value = scope.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if value > 0:
                label = "config" if scope_idx == 0 else "config.text_config"
                return int(value), f"{label}.{key}"
    return None


def resolve_routing_factor(
    config: Mapping[str, Any],
    *,
    observed_expert_counts: Mapping[str, int] | None = None,
    context: str = "read_traffic",
) -> RoutingFactor:
    """Derive ``topk`` and ``E`` from the model config, and cross-check ``E``.

    Both terms are *declarations*, never guesses: an MoE model whose config
    does not state how many experts a token activates raises.  The
    ``moe_imatrix`` fallback of "assume 8" is exactly the heuristic principle
    2 forbids here, because it would silently mis-price the single largest
    term in the ledger.

    ``observed_expert_counts`` maps a tensor name to the stack depth actually
    measured for it (``stats['num_experts']`` pre-export, ``shape[0]`` on an
    exported 3-D stack).  Any disagreement with the config is a hard error:
    one of the two is describing a different model.
    """
    scopes = _config_scopes(config)
    topk = _first_positive_int(scopes, _EXPERTS_PER_TOK_KEYS)
    experts = _first_positive_int(scopes, _EXPERT_COUNT_KEYS)
    if topk is None or experts is None:
        missing = []
        if topk is None:
            missing.append(f"one of {list(_EXPERTS_PER_TOK_KEYS)}")
        if experts is None:
            missing.append(f"one of {list(_EXPERT_COUNT_KEYS)}")
        raise ReadTrafficError(
            f"[read_traffic] {context}: this checkpoint carries routed-expert "
            f"tensors but its config declares no {' and no '.join(missing)}. "
            "The routed read probability is topk/E and both terms must be "
            "declared -- refusing to assume a default, which would mis-price "
            "the largest term in the ledger."
        )
    topk_value, topk_source = topk
    experts_value, experts_source = experts
    if topk_value > experts_value:
        raise ReadTrafficError(
            f"[read_traffic] {context}: config declares "
            f"{topk_source}={topk_value} > {experts_source}={experts_value}; "
            "a token cannot activate more experts than exist."
        )
    exactness = "exact_under_per_layer_uniform_expert_stacks"
    for name, observed in sorted((observed_expert_counts or {}).items()):
        if int(observed) != experts_value:
            raise ReadTrafficError(
                f"[read_traffic] {context}: {name} carries {int(observed)} "
                f"experts but {experts_source}={experts_value}. The read "
                "probability topk/E is only meaningful when the config and "
                "the tensors describe the same model."
            )
    return RoutingFactor(
        num_experts_per_tok=topk_value,
        n_routed_experts=experts_value,
        source=f"{topk_source} / {experts_source}",
        exactness=exactness,
    )


def read_model_config(model_path: str | os.PathLike) -> dict:
    path = Path(model_path) / "config.json"
    if not path.is_file():
        raise ReadTrafficError(
            f"[read_traffic] no config.json under {str(model_path)!r}; the "
            "routed-expert read probability cannot be derived without the "
            "architecture's own expert declarations."
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _has_experts_segment(name: str) -> bool:
    """True when a qname structurally sits under a routed-expert container.

    ``experts`` must be a whole path SEGMENT, which is what keeps a shared
    expert (``mlp.shared_experts.gate_proj``) out: its segment is
    ``shared_experts``, and a shared expert is read on every token.  This is
    the same structural test ``ModelProfile._packed_expert_projection_leaf``
    and ``footprint.packed_expert_alias`` already use.
    """
    return "experts" in str(name).split(".")


def _declares_routed_moe(profile) -> bool:
    """True when this profile declares a routed-expert MoE structure."""
    for accessor, empty in (
        ("per_expert_moe_regex", None),
        ("packed_expert_param_names", frozenset()),
        ("unpacked_expert_projection_names", ()),
    ):
        try:
            value = getattr(profile, accessor)()
        except Exception:
            continue
        if value and value != empty:
            return True
    return False


def _mtp_prefixes(profile) -> tuple[str, ...]:
    """Every spelling under which this profile's MTP sidecar can appear."""
    out: list[str] = []
    for value in (
        getattr(profile, "mtp_source_prefix", lambda: None)(),
        getattr(profile, "mtp_layer_prefix", lambda: None)(),
    ):
        if not value:
            continue
        text = str(value)
        out.append(text if text.endswith(".") else text + ".")
    # A recipe spells the sidecar `model.mtp.` where the checkpoint spells it
    # `mtp.`; both reach this classifier, so both are declared.
    for text in tuple(out):
        out.append("model." + text)
    return tuple(dict.fromkeys(out))


def resolve_vocab_size(config: Mapping[str, Any] | None) -> int | None:
    """The declared vocabulary size, or ``None`` if the config states none."""
    found = _first_positive_int(_config_scopes(config or {}), _VOCAB_SIZE_KEYS)
    return None if found is None else found[0]


def _module_stem(name: str) -> str:
    """The module a tensor hangs off: its name minus the final leaf."""
    base = strip_weight_leaf(str(name))
    for suffix in fp._SIDECAR_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            return base
    head, _, _leaf = base.rpartition(".")
    return head or base


def scaled_module_stems(
    header_meta: Mapping[str, tuple[str, tuple[int, ...]]],
) -> frozenset[str]:
    """Module stems that ship a float **scale** sidecar in this checkpoint.

    A scale is what turns packed integers into numbers, so a stem carrying one
    holds a quantized *operand*, not an index table.  The suffixes come from
    ``footprint._SIDECAR_SUFFIXES`` -- the same producer-side constant the byte
    partition already folds on -- rather than from a name guess invented here.
    """
    out: set[str] = set()
    for name, (dtype, _shape) in header_meta.items():
        if str(dtype).upper() in INTEGER_SAFETENSORS_DTYPES:
            continue  # a scale is float; an integer sidecar is not one
        if any(str(name).endswith(s) for s in fp._SIDECAR_SUFFIXES):
            out.add(_module_stem(name))
    return frozenset(out)


def quantization_targets(model_path: str) -> tuple[str, ...]:
    """The module patterns this artifact declares as quantized, if any.

    ``config_groups[*].targets`` in ``quant_config.json`` (CB lane) or in
    ``config.json``'s ``quantization_config`` (compressed-tensors) is the
    artifact's own statement of which modules hold packed weights.  It is
    needed because the scale-sidecar test cannot see the CB lane: a CB payload
    keeps its scales in the codebook sidecar, so ``lm_head.cb_qweight`` has no
    scale tensor in the shard set at all -- but the artifact does declare
    ``targets: ["lm_head"]``.
    """
    payloads: list[Mapping[str, Any]] = []
    for filename, key in (("quant_config.json", None),
                          ("config.json", "quantization_config")):
        path = Path(model_path) / filename
        if not path.is_file():
            continue
        try:
            loaded = json.loads(path.read_text())
        except Exception:
            continue
        section = loaded if key is None else loaded.get(key)
        if isinstance(section, Mapping):
            payloads.append(section)
    out: list[str] = []
    for section in payloads:
        groups = section.get("config_groups")
        if not isinstance(groups, Mapping):
            continue
        for group in groups.values():
            if not isinstance(group, Mapping):
                continue
            for target in group.get("targets") or ():
                out.append(str(target))
    return tuple(dict.fromkeys(out))


def _matches_quant_target(stem: str, targets: Iterable[str]) -> bool:
    """compressed-tensors target semantics: a literal name or an ``re:``."""
    for target in targets:
        if target.startswith("re:"):
            try:
                if re.fullmatch(target[3:], stem):
                    return True
            except re.error:
                continue
        elif target == stem or stem.endswith("." + target):
            return True
    return False


def is_indexed_lookup(
    dtype: str | None,
    shape: tuple[int, ...] | None,
    vocab_size: int | None,
    *,
    name: str | None = None,
    scaled_stems: frozenset[str] | None = None,
    quant_targets: tuple[str, ...] = (),
) -> bool:
    """Is this tensor a per-token *indexed lookup* rather than an operand?

    Three facts, all read off the safetensors headers and the model config,
    and **all required**:

    1. the dtype is an integer type, so the tensor cannot be a float matmul
       operand *on its own*;
    2. its leading axis is the vocabulary, so a decode step addresses one row
       of it by token id -- exactly the reason the embedding is excluded; and
    3. its module is **not declared a quantized weight** -- neither by a float
       scale sidecar (nothing turns those integers into numbers) nor by the
       artifact's own ``config_groups[*].targets``.

    Each of the first two was measured to be insufficient alone, on real
    artifacts:

    * dtype only -- the packed weight payload of both quantized lanes is
      ``U8`` (81.65 GB of ``cb_qweight`` on the shipped DSv4 CB body, 15.7 GB
      of NVFP4 ``weight_packed`` on Ornith-1.5-35B), streamed in full.  A
      dtype-only rule excludes 94% of that artifact.
    * dtype + vocabulary axis -- a **quantized ``lm_head``** is integer-dtyped
      *and* vocabulary-keyed, and it is streamed in full.  On the
      `embed-smoke` CB-head export that two-fact rule dropped 857,736 B of
      real logits traffic, an **under**-count: the dangerous direction.
    * the scale-sidecar half of fact 3 alone -- a CB payload keeps its scales
      in the codebook sidecar, so ``lm_head.cb_qweight`` has no scale tensor
      in the shard set; the artifact's declared ``targets`` is what covers it.

    What survives all three is DSv4's ``ffn.gate.tid2eid`` -- I64
    ``[129280, 6]``, token id -> its six expert ids, no scale anywhere on
    ``ffn.gate`` and no quantization target naming it -- which is the case the
    rule was derived from.

    The residual is deliberate and one-directional: an integer table keyed by
    anything other than the vocabulary, or one that happens to sit under a
    quantized module, stays in ``held_fixed`` at ``p=1`` and **over**-counts.
    """
    if not dtype or str(dtype).upper() not in INTEGER_SAFETENSORS_DTYPES:
        return False
    if not shape or vocab_size is None:
        return False
    if int(shape[0]) != int(vocab_size):
        return False
    if scaled_stems is None or name is None:
        # Fact 3 cannot be established, so the tensor is not classified as a
        # lookup: it stays at p=1 and over-counts, never the reverse.
        return False
    stem = _module_stem(name)
    if stem in scaled_stems:
        return False
    return not _matches_quant_target(stem, quant_targets)


_TIE_KEYS = ("tie_word_embeddings", "tie_embeddings", "tie_embedding")


@dataclass(frozen=True)
class EmbeddingDisposition:
    """Whether this checkpoint's embedding table is streamed per token."""

    streamed: bool
    lm_head_present: bool
    tie_declared: bool | None
    reason: str

    @property
    def read_class(self) -> str:
        return "held_fixed" if self.streamed else "excluded_embedding"

    def as_dict(self) -> dict[str, Any]:
        return {
            "streamed_per_token": self.streamed,
            "read_class": self.read_class,
            "lm_head_tensor_present": self.lm_head_present,
            "config_tie_word_embeddings": self.tie_declared,
            "reason": self.reason,
        }


def _matches_declared(base: str, declared: str) -> bool:
    """``base`` is the tensor the profile means by ``declared``."""
    declared = strip_weight_leaf(str(declared))
    return base == declared or base.endswith("." + declared)


def _declared_tie(config: Mapping[str, Any]) -> bool | None:
    for scope in _config_scopes(config or {}):
        for key in _TIE_KEYS:
            value = scope.get(key)
            if isinstance(value, bool):
                return value
    return None


def resolve_embedding_disposition(
    names: Iterable[str],
    *,
    profile,
    config: Mapping[str, Any] | None = None,
    context: str = "read_traffic",
) -> EmbeddingDisposition:
    """Decide whether the embedding table is per-token read traffic.

    The invariant this rests on is that the **logits projection is streamed
    exactly once per generated token**.  So the question is not "is this
    model tied?" as a config fact but "does this checkpoint carry a separate
    output projection?" as an observed one:

    * a separate ``lm_head`` tensor exists -> it carries the p=1 logits
      traffic and the embedding table is gathered only, ``p = 0``;
    * no output projection tensor exists -> the embedding table *is* the
      output projection and is streamed in full, ``p = 1``.

    ``config['tie_word_embeddings']`` is a cross-check rather than the
    decision: a config declaring the model untied while the checkpoint ships
    no output projection is a contradiction, and one of the two descriptions
    is of a different model.  Deciding by observation is also what keeps this
    correct for a config that simply omits the key (transformers defaults it
    to ``True``, which is exactly the kind of implicit default principle 2
    forbids relying on).

    ``names`` must carry **both** spellings of every tensor -- the checkpoint
    key and the live name -- because the two declarations this reads are in
    different namespaces: ``lm_head_name()`` is the checkpoint spelling (DSv4
    says ``head``) while ``embedding_name()`` is the live one (``model.
    embed_tokens``, where the checkpoint says ``embed``).
    """
    lm_head = strip_weight_leaf(str(profile.lm_head_name()))
    embedding = strip_weight_leaf(str(profile.embedding_name()))
    lm_head_present = False
    embedding_present = False
    for raw in names:
        base = strip_weight_leaf(str(raw))
        if _matches_declared(base, lm_head):
            lm_head_present = True
        elif _matches_declared(base, embedding):
            embedding_present = True
    tie = _declared_tie(config or {})

    if lm_head_present:
        reason = (
            f"a separate output projection ({lm_head!r}) carries the "
            "per-token logits traffic, so the embedding table is gathered "
            "one row at a time"
        )
        if tie is True:
            reason += (
                "; config declares tie_word_embeddings=true and the "
                "checkpoint materializes lm_head anyway -- the projection is "
                "still streamed exactly once, counted there"
            )
        return EmbeddingDisposition(False, True, tie, reason)

    if not embedding_present:
        # Neither tensor is here (a body-only shard set, a sub-model). There
        # is no embedding byte to classify either way.
        return EmbeddingDisposition(
            False, False, tie,
            "this checkpoint carries neither an embedding table nor an "
            "output projection",
        )

    if tie is False:
        raise ReadTrafficError(
            f"[read_traffic] {context}: the config declares "
            f"tie_word_embeddings=false but no {lm_head!r} tensor exists in "
            "this checkpoint, so nothing carries the per-token logits "
            "traffic. Either the profile names the output projection wrongly "
            "or the config describes a different model; refusing to guess "
            "whether the embedding table is streamed."
        )
    return EmbeddingDisposition(
        True, False, tie,
        "no separate output projection exists, so the embedding table IS the "
        "logits projection and the decode streams all of it every token"
        + ("" if tie else " (config omits tie_word_embeddings; decided by "
                          "the absence of an output projection tensor)"),
    )


def classify_read_class(
    name: str,
    *,
    profile,
    checkpoint_key: str | None = None,
    in_assignment: bool = False,
    embedding_streamed: bool | None = None,
    dtype: str | None = None,
    shape: tuple[int, ...] | None = None,
    vocab_size: int | None = None,
    scaled_stems: frozenset[str] | None = None,
    quant_targets: tuple[str, ...] = (),
    context: str = "read_traffic",
    projection: NameProjection | None = None,
) -> str:
    """The read class of one tensor.  See :data:`READ_CLASS_TABLE`.

    ``name`` is the live/allocator spelling; ``checkpoint_key`` is the source
    spelling when they differ (the MTP sidecar is the case that matters — the
    profile's ``checkpoint_to_live_name`` declines it, so the manifest falls
    back to the raw key).  Both spellings are tested against every rule, so a
    class is never missed on naming alone (project memory: a Linear has three
    names).  The checkpoint→live bridge is the shared
    :class:`~prismaquant.name_projection.NameProjection`, not a private
    mapping here; pass the caller's instance (``projection=``) so every span
    shares one projection.

    ``embedding_streamed`` answers the one question a single tensor's name
    cannot: a tied embedding IS the output projection and is read in full
    every token.  It is only consulted for the embedding tensor itself, and
    only :func:`resolve_embedding_disposition` may answer it — passing
    nothing raises there rather than assuming untied.

    Raises rather than falling through when a name is structurally a routed
    expert but the profile cannot name its role: that is an undeclared
    architecture, and pricing it at ``p=1`` would silently inflate the
    ledger by the whole expert mass.
    """
    spellings = {str(name)}
    if checkpoint_key:
        spellings.add(str(checkpoint_key))
    bases = {strip_weight_leaf(s) for s in spellings}

    if any(base.startswith(CODEBOOK_TENSOR_PREFIX) for base in bases):
        return "resident_codebooks"

    mtp_prefixes = _mtp_prefixes(profile)
    if any(base.startswith(p) for base in bases for p in mtp_prefixes):
        return "excluded_mtp"

    # Only the floor half passes a dtype: a unit the allocator assigned is a
    # matmul operand by construction, and its stored bytes are a rendered
    # weight format rather than the source dtype.  (An NVFP4-quantized SOURCE
    # would present `lm_head.weight_packed` as U8 with a vocabulary-length
    # leading axis, and it is a weight, not a lookup.)
    if not in_assignment and is_indexed_lookup(
        dtype, shape, vocab_size,
        name=checkpoint_key or name, scaled_stems=scaled_stems,
        quant_targets=quant_targets,
    ):
        return "excluded_indexed_lookup"

    if any(_has_experts_segment(base) for base in bases):
        # The `experts` path SEGMENT is the structural fact, and it is the
        # same fact in all three namespaces (recipe / checkpoint / vLLM), so
        # it -- not a leaf-name table -- is what decides the read class. All
        # routed roles share one read probability, so naming the role would
        # add nothing here; what must be declared is that this architecture
        # HAS a routed MoE at all, and a name under `experts.` on a profile
        # that declares none is a contradiction, not a default.
        if not _declares_routed_moe(profile):
            raise ReadTrafficError(
                f"[read_traffic] {context}: {name!r} sits under a routed-"
                "expert container but this model profile declares no routed "
                "MoE structure (no per-expert regex, no packed-expert "
                "params, no unpacked expert projections), so its read "
                "probability is undeclared. Pricing it as always-active "
                "would inflate expected read bytes by the entire expert mass."
            )
        return "routed_experts"

    embedding = strip_weight_leaf(str(profile.embedding_name()))
    if any(_matches_declared(base, embedding) for base in bases):
        if embedding_streamed is None:
            raise ReadTrafficError(
                f"[read_traffic] {context}: {name!r} is this profile's "
                "embedding table, whose read probability depends on whether "
                "it is also the output projection (tie_word_embeddings). The "
                "caller did not resolve that; call "
                "resolve_embedding_disposition() over the checkpoint's tensor "
                "names and pass embedding_streamed=. Assuming untied would "
                "silently drop the whole logits projection from the ledger "
                "on every tied model."
            )
        return "held_fixed" if embedding_streamed else "excluded_embedding"

    # The profile declining to map a checkpoint key into the live graph is
    # the architecture's OWN declaration that the tensor is not part of the
    # text decode path (vision/audio towers).  It is a declaration, not a
    # name test, which is why it is the rule rather than a prefix list.
    # Routed through the shared name-projection layer: the DECLARED drop is
    # data (DECLARED_OUT_OF_GRAPH), and an accessor that fails raises
    # NameProjectionError instead of silently passing the key through.
    if checkpoint_key is not None:
        if projection is None:
            projection = NameProjection(profile)
        projected = projection.checkpoint_to_live(str(checkpoint_key))
        if projected.outcome == DECLARED_OUT_OF_GRAPH:
            return "excluded_non_text_graph"

    return "dense" if in_assignment else "held_fixed"


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def _header_meta(model_path: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    """``{checkpoint key: (dtype, shape)}`` from the shards' own headers."""
    out: dict[str, tuple[str, tuple[int, ...]]] = {}
    for shard in sorted(Path(model_path).glob("*.safetensors")):
        for name, meta in fp._read_safetensors_header(str(shard)).items():
            if name == "__metadata__":
                continue
            out[name] = (
                str(meta.get("dtype") or ""),
                tuple(int(d) for d in (meta.get("shape") or ())),
            )
    return out


class _IntegerTensorTally:
    """How the integer-dtype tensors in a checkpoint were resolved.

    Published with the figure because the rule's residual is an over-count:
    an integer tensor that is *not* vocabulary-keyed is read at ``p=1``, and
    a reviewer must be able to see how many bytes that is (on a quantized
    artifact it is the whole packed weight payload, by design).
    """

    def __init__(self, vocab_size: int | None) -> None:
        self.vocab_size = vocab_size
        self.excluded_bytes = 0
        self.excluded_tensors = 0
        self.read_in_full_bytes = 0
        self.read_in_full_tensors = 0

    def note(self, dtype: str | None, shape, nbytes: int, klass: str) -> None:
        if not dtype or str(dtype).upper() not in INTEGER_SAFETENSORS_DTYPES:
            return
        if klass == "excluded_indexed_lookup":
            self.excluded_bytes += int(nbytes)
            self.excluded_tensors += 1
        elif klass in STREAMED_CLASSES:
            self.read_in_full_bytes += int(nbytes)
            self.read_in_full_tensors += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": (
                "integer dtype AND shape[0] == vocab_size AND the module is "
                "not declared a quantized weight (no float scale sidecar and "
                "no config_groups[*].targets match)"
            ),
            "vocab_size": self.vocab_size,
            "excluded_bytes": self.excluded_bytes,
            "excluded_tensors": self.excluded_tensors,
            "integer_bytes_read_in_full": self.read_in_full_bytes,
            "integer_tensors_read_in_full": self.read_in_full_tensors,
            "note": (
                "integer dtype alone does not mean lookup -- the packed "
                "weight payload of both quantized lanes is U8 and is streamed "
                "in full, and a quantized lm_head is vocabulary-keyed too, so "
                "what separates an index from a number is that no module "
                "declares the tensor a quantized weight -- neither a float "
                "scale sidecar next to it nor a config_groups[*].targets entry "
                "naming it (a CB lm_head ships no scale sidecar at all, so "
                "both halves of that fact are load-bearing); anything the "
                "three facts cannot establish stays at p=1, which over-counts "
                "rather than under-counts"
            ),
        }


def _new_class_totals() -> dict[str, dict[str, Any]]:
    return {
        name: {"stored_bytes": 0, "read_bytes": 0.0, "n_tensors": 0}
        for name in READ_CLASS_TABLE
    }


def _finalize(
    class_totals: Mapping[str, Mapping[str, Any]],
    routing: RoutingFactor | None,
    *,
    reconciliation: Mapping[str, Any],
    embedding: EmbeddingDisposition,
    integer_tally: _IntegerTensorTally,
    unpriced_assignment_names: tuple[str, ...] = (),
    measured_from: str,
) -> dict:
    read_bytes = sum(
        float(class_totals[name]["read_bytes"]) for name in STREAMED_CLASSES)
    breakdown = {
        "dense": int(round(class_totals["dense"]["read_bytes"])),
        "routed": int(round(class_totals["routed_experts"]["read_bytes"])),
        "held_fixed": int(round(class_totals["held_fixed"]["read_bytes"])),
        "resident_codebooks": int(
            class_totals["resident_codebooks"]["stored_bytes"]),
    }
    return {
        "schema": SCHEMA,
        "read_bytes_per_token": int(round(read_bytes)),
        "read_gb_per_token": read_bytes / fp.GB,
        "scope": READ_SCOPE,
        "measured_from": measured_from,
        "breakdown": breakdown,
        "excluded": {
            "embedding_bytes": int(
                class_totals["excluded_embedding"]["stored_bytes"]),
            "indexed_lookup_bytes": int(
                class_totals["excluded_indexed_lookup"]["stored_bytes"]),
            "mtp_bytes": int(class_totals["excluded_mtp"]["stored_bytes"]),
            "non_text_graph_bytes": int(
                class_totals["excluded_non_text_graph"]["stored_bytes"]),
            "note": (
                "real bytes with zero batch-1 text-decode traffic; indexed "
                "lookups are addressed one row at a time, and the MTP sidecar "
                "becomes per-token traffic under spec-decode and can be added "
                "back from this figure exactly"
            ),
        },
        "embedding": embedding.as_dict(),
        "indexed_lookups": integer_tally.as_dict(),
        "routing": (
            {
                "num_experts_per_tok": routing.num_experts_per_tok,
                "n_routed_experts": routing.n_routed_experts,
                "read_probability": routing.read_probability,
                "source": routing.source,
                "exactness": routing.exactness,
            }
            if routing is not None
            else {"read_probability": None, "source": "no routed experts"}
        ),
        "classes": {
            name: {
                "stored_bytes": int(entry["stored_bytes"]),
                "read_bytes": int(round(float(entry["read_bytes"]))),
                "n_tensors": int(entry["n_tensors"]),
                "read_probability": (
                    routing.read_probability
                    if name == "routed_experts" and routing is not None
                    else READ_CLASS_TABLE[name]
                ),
            }
            for name, entry in class_totals.items()
        },
        "reconciliation": dict(reconciliation),
        "unpriced_assignment_names": list(unpriced_assignment_names),
    }


def assignment_read_traffic(
    assignment: Mapping[str, str],
    stats: Mapping[str, dict],
    *,
    model_path: str | os.PathLike,
    profile=None,
    source_manifest: Mapping[str, int] | None = None,
    source_total_bytes: int | None = None,
    config: Mapping[str, Any] | None = None,
    cb_serialization_context=None,
    per_expert_assignment: Mapping[str, str] | None = None,
    context: str = "assignment_read_traffic",
) -> dict:
    """Expected per-token decode read bytes for ``assignment``, both lanes.

    ``assignment`` is the allocator's *expanded*, post-promotion per-Linear
    recipe (fused-sibling and packed-MoE coupling already reflected) and
    ``stats`` the probe's per-Linear stats, exactly as
    :func:`footprint.assignment_artifact_bytes` takes them -- and that
    function is this one's byte authority.  Stored bytes for an assigned unit
    come from :func:`footprint.format_tensor_payload_breakdown` (the shared
    per-unit primitive that already prices NVFP4 group scales, FP8 row
    scales, and CB index/row-scale/layout bytes); stored bytes for every
    tensor the allocator never decided come from the checkpoint's own
    safetensors spans via :func:`footprint.source_tensor_span_bytes`.

    The two halves are then **reconciled against the whole-assignment
    footprint before any read probability is applied**, and a mismatch of a
    single byte raises.  That is what makes this a re-use of the footprint
    accounting rather than a second copy of it: the ledger cannot drift
    without failing.

    Returns a dict with ``read_bytes_per_token`` / ``read_gb_per_token``, the
    four-key ``breakdown`` (dense / routed / held_fixed / resident_codebooks),
    the itemized ``excluded`` bytes, the ``routing`` factor and its
    provenance, per-class totals, and the ``reconciliation`` block.  CB
    assignments require ``cb_serialization_context`` for the same reason
    ``assignment_artifact_bytes`` does.
    """
    model_path = str(model_path)
    if profile is None:
        from prismaquant.model_profiles import detect_profile_with_warning
        profile = detect_profile_with_warning(
            model_path, entrypoint="read-traffic")
    if config is None:
        config = read_model_config(model_path)
    measured_total, by_dtype = fp.source_checkpoint_bytes(model_path)
    regime = fp.source_regime(by_dtype)
    if source_total_bytes is None:
        source_total_bytes = measured_total
    if source_manifest is None:
        source_manifest = fp.source_tensor_bytes_manifest(
            model_path,
            profile.checkpoint_to_live_name,
            profile.packed_expert_parent_for_projection,
        )

    # --- byte authority -----------------------------------------------------
    totals = fp.assignment_artifact_bytes(
        assignment, stats,
        source_total_bytes=int(source_total_bytes),
        source_manifest=source_manifest,
        regime=regime,
        context=context,
        cb_serialization_context=cb_serialization_context,
        per_expert_assignment=per_expert_assignment,
    )

    merged = dict(assignment)
    if per_expert_assignment:
        merged.update(per_expert_assignment)
    unpriced = tuple(totals["missing_stats_names"])
    grouped_payload = totals.get("per_expert_format_group_payload") or {}
    grouped_qnames = {
        qname
        for group in (grouped_payload.get("groups") or {}).values()
        if is_cb_format(group["format"])
        for qname in group["member_qnames"]
    }
    cb_payload = totals.get("cb_serialized_payload") or {}
    cb_per_tensor = cb_payload.get("per_tensor") or {}

    # Mirrors footprint's own passthrough resolution exactly: the same names,
    # resolved by the same helper, so a passthrough unit is charged the
    # measured source span rather than a closed form that might disagree.
    passthrough_names = [
        qname for qname, raw in merged.items()
        if fr.canonical_format_name(raw) in _source_passthrough_formats()
    ]
    passthrough_spans = (
        fp.resolve_reencoded_source_bytes(
            source_manifest, passthrough_names, context=context)
        if passthrough_names else {}
    )

    # One shared name-projection for every checkpoint→live question below.
    # Built from the same profile the byte authorities take, so the ledger
    # cannot describe a different model than the footprint does.
    projection = NameProjection(profile)

    # The embedding's read probability is a whole-checkpoint fact (is there a
    # separate output projection?), so it is resolved once, over every name in
    # the artifact, before any tensor is classified.
    span_bytes = fp.source_tensor_span_bytes(model_path)
    live_names: list[str] = list(merged)

    def _live(ckpt_key: str) -> str:
        # The layer's declared drop (vision/audio/MTP/scale keys) keeps the
        # raw checkpoint spelling in play as a NAME for rule matching -- which
        # is what the pre-projection code did with a ``None`` mapping too --
        # while a failing profile accessor now refuses loudly through
        # NameProjectionError instead of silently passing the key through.
        projected = projection.checkpoint_to_live(ckpt_key)
        if projected.outcome == MAPPED:
            return projected.target
        return strip_weight_leaf(ckpt_key)

    for key in span_bytes:
        live_names.append(key)        # the checkpoint spelling (`head`)
        live_names.append(_live(key))  # and the live one (`lm_head`)
    embedding = resolve_embedding_disposition(
        live_names, profile=profile, config=config, context=context)
    vocab_size = resolve_vocab_size(config)
    header_meta = _header_meta(model_path)
    scaled_stems = scaled_module_stems(header_meta)
    quant_targets = quantization_targets(model_path)
    integer_tally = _IntegerTensorTally(vocab_size)

    class_totals = _new_class_totals()
    observed_expert_counts: dict[str, int] = {}
    priced_names: list[str] = []
    saw_routed = False

    def _charge(nbytes: int, klass: str) -> None:
        entry = class_totals[klass]
        entry["stored_bytes"] += int(nbytes)
        entry["n_tensors"] += 1

    # --- half one: the units the allocator decided ---------------------------
    for qname, raw_format in merged.items():
        entry = stats.get(qname)
        if entry is None and qname.endswith(".weight"):
            entry = stats.get(strip_weight_leaf(qname))
        if not isinstance(entry, dict):
            continue  # unpriced: its source bytes stay in the floor half
        # `priced` in footprint's own loop: these are the names whose SOURCE
        # spans it removes from the floor, so they are the names whose spans
        # this ledger must treat as already covered.
        priced_names.append(qname)
        shape = _shape_from_stats(entry)
        name = fr.canonical_format_name(raw_format)
        if qname in grouped_qnames:
            continue  # priced once per physical CB sub-stack, below
        if qname in passthrough_spans:
            nbytes = int(passthrough_spans[qname])
        elif is_cb_format(name):
            item = cb_per_tensor.get(qname)
            if item is None:
                raise ReadTrafficError(
                    f"[read_traffic] {context}: {qname!r} is assigned the CB "
                    f"format {name} but the whole-assignment CB payload has "
                    "no entry for it, so its stored bytes are unknown. "
                    "Refusing to contribute a silent zero."
                )
            nbytes = int(item["tensor_payload_bytes"])
        else:
            nbytes = int(fp.format_tensor_payload_breakdown(
                name, shape, qname=qname,
                cb_serialization_context=cb_serialization_context,
            )["tensor_payload_bytes"])
        if name == "NVFP4":
            nbytes += int(fp.nvfp4_global_sidecar_bytes(
                qname, shape,
                weight_only=bool(
                    entry.get(fp.NVFP4_WEIGHT_ONLY_STATS_KEY, False)),
            ))
        klass = classify_read_class(
            qname, profile=profile, in_assignment=True,
            embedding_streamed=embedding.streamed, context=context)
        if klass == "routed_experts":
            saw_routed = True
            if len(shape) == 3:
                observed_expert_counts[qname] = int(shape[0])
        _charge(nbytes, klass)

    # CB split sub-stacks are physical tensors of their own; charge each once
    # to the class of its members (all members of a group share a role).
    for key, group in sorted((grouped_payload.get("groups") or {}).items()):
        members = group["member_qnames"]
        if not members:
            raise ReadTrafficError(
                f"[read_traffic] {context}: per-expert group {key!r} declares "
                "no member tensors, so it cannot be classified.")
        klass = classify_read_class(
            members[0], profile=profile, in_assignment=True,
            embedding_streamed=embedding.streamed, context=context)
        saw_routed |= klass == "routed_experts"
        _charge(int(group["tensor_payload_bytes"]), klass)
        if int(group["codebook_sidecar_bytes"]):
            _charge(int(group["codebook_sidecar_bytes"]), "resident_codebooks")

    cb_sidecar_bytes = int(cb_payload.get("codebook_sidecar_bytes") or 0)
    if cb_sidecar_bytes:
        _charge(cb_sidecar_bytes, "resident_codebooks")

    # --- half two: every tensor the allocator never decided ------------------
    covered_spans: set[str] = set()
    span_map = getattr(source_manifest, "spans", {}) or {}
    for qname in priced_names:
        key = qname if qname in source_manifest else strip_weight_leaf(qname)
        covered_spans.update(span_map.get(key, ()))

    for ckpt_key, nbytes in sorted(span_bytes.items()):
        if fp.source_span_identity(ckpt_key) in covered_spans:
            continue
        dtype, shape = header_meta.get(ckpt_key, (None, None))
        klass = classify_read_class(
            _live(ckpt_key), profile=profile, checkpoint_key=ckpt_key,
            in_assignment=False, embedding_streamed=embedding.streamed,
            dtype=dtype, shape=shape, vocab_size=vocab_size,
            scaled_stems=scaled_stems, quant_targets=quant_targets,
            context=context, projection=projection)
        saw_routed |= klass == "routed_experts"
        integer_tally.note(dtype, shape, nbytes, klass)
        _charge(int(nbytes), klass)

    # --- reconcile before weighting -----------------------------------------
    ledger_total = sum(
        int(entry["stored_bytes"]) for entry in class_totals.values())
    expected = int(totals["artifact_payload_bytes"])
    if ledger_total != expected:
        raise ReadTrafficError(
            f"[read_traffic] {context}: the per-tensor read ledger totals "
            f"{ledger_total} bytes but footprint.assignment_artifact_bytes "
            f"prices the same assignment at {expected} bytes (delta "
            f"{ledger_total - expected}). One of the two is wrong about this "
            "artifact and neither number may be published. The ledger must "
            "partition exactly the bytes the export ships, or the read-bytes "
            "figure is a different artifact's."
        )

    routing = None
    if saw_routed:
        routing = resolve_routing_factor(
            config,
            observed_expert_counts=observed_expert_counts,
            context=context,
        )
    _apply_probabilities(class_totals, routing)
    return _finalize(
        class_totals, routing,
        reconciliation={
            "ledger_stored_bytes": ledger_total,
            "footprint_artifact_payload_bytes": expected,
            "agrees": True,
            "source_total_bytes": int(source_total_bytes),
            "n_priced_units": len(priced_names),
        },
        embedding=embedding,
        integer_tally=integer_tally,
        unpriced_assignment_names=unpriced,
        measured_from="allocator assignment + source checkpoint spans",
    )


def _apply_probabilities(
    class_totals: dict[str, dict[str, Any]],
    routing: RoutingFactor | None,
) -> None:
    for name, entry in class_totals.items():
        if name == "routed_experts":
            p = routing.read_probability if routing is not None else 0.0
        else:
            p = READ_CLASS_TABLE[name] or 0.0
        entry["read_bytes"] = float(entry["stored_bytes"]) * float(p)


def _source_passthrough_formats() -> frozenset[str]:
    from prismaquant.allocator_candidates import SOURCE_PASSTHROUGH_FORMATS
    return frozenset(SOURCE_PASSTHROUGH_FORMATS)


# ---------------------------------------------------------------------------
# Post-export: measure the artifact that was actually written
# ---------------------------------------------------------------------------

def _exported_codebook_sidecar_bytes(
    export_dir: str, *, context: str,
) -> tuple[int, str | None]:
    """Resident bytes of a CB artifact's codebook sidecar, if it declares one.

    The CB lane ships its codebooks in a globbed sidecar file rather than in
    the shard set (so vLLM's weight loader never sees them), which means the
    safetensors ledger above cannot see them either.  Reporting ``0`` resident
    codebook bytes for an artifact that ships 748 KB of them would be exactly
    the silent zero this module refuses everywhere else, so the file is read
    from the artifact's OWN ``quant_config.json`` declaration
    (``codebook_file``) -- not from a glob, and not from a guess about naming.
    """
    config_path = Path(export_dir) / "quant_config.json"
    if not config_path.is_file():
        return 0, None
    try:
        declared = json.loads(config_path.read_text()).get("codebook_file")
    except Exception:
        return 0, None
    if not declared:
        return 0, None
    path = Path(export_dir) / str(declared)
    if not path.is_file():
        raise ReadTrafficError(
            f"[read_traffic] {context}: quant_config.json declares "
            f"codebook_file={declared!r} but no such file exists under "
            f"{export_dir!r}. Its resident bytes cannot be reported as zero: "
            "the artifact is either incomplete or mis-declared."
        )
    try:
        header = fp._read_safetensors_header(str(path))
        nbytes = sum(
            int(meta["data_offsets"][1]) - int(meta["data_offsets"][0])
            for name, meta in header.items() if name != "__metadata__"
        )
        return nbytes, f"{declared} (tensor data)"
    except ReadTrafficError:
        raise
    except Exception:
        return path.stat().st_size, f"{declared} (file bytes)"


def exported_checkpoint_read_traffic(
    export_dir: str | os.PathLike,
    *,
    profile=None,
    config: Mapping[str, Any] | None = None,
    context: str = "exported_checkpoint_read_traffic",
) -> dict:
    """The same stat, measured from an exported checkpoint's own headers.

    This is the form the shipcard stamps, and it is the stronger of the two:
    it reads the bytes the artifact actually ships rather than the bytes a
    recipe predicts, so it cannot describe a different assignment than the
    one on disk -- the failure mode that has now shipped twice on
    ``achieved_bpp`` (see :func:`shipcard.allocator_achieved_bpp`).

    Every tensor span in every shard is classified and counted exactly once;
    the sum of stored bytes equals the checkpoint's own tensor-data total by
    construction, which is asserted.
    """
    export_dir = str(export_dir)
    if profile is None:
        from prismaquant.model_profiles import detect_profile_with_warning
        profile = detect_profile_with_warning(
            export_dir, entrypoint="read-traffic")
    if config is None:
        config = read_model_config(export_dir)

    projection = NameProjection(profile)
    spans = fp.source_tensor_span_bytes(export_dir)

    # Classify on the LIVE spelling with the on-disk one alongside: a
    # multimodal checkpoint stores the embedding as
    # `model.language_model.embed_tokens.weight`, and only the profile's own
    # mapping turns that into the name the profile declares.  The mapping is
    # the shared name-projection layer: a declared drop keeps the raw key as a
    # rule-matching name (as the pre-projection code did with `None`), while
    # a failing profile accessor refuses instead of passing through.
    def _live(ckpt_key: str) -> str:
        projected = projection.checkpoint_to_live(ckpt_key)
        if projected.outcome == MAPPED:
            return projected.target
        return strip_weight_leaf(ckpt_key)

    embedding = resolve_embedding_disposition(
        [name for key in spans for name in (key, _live(key))],
        profile=profile, config=config, context=context,
    )

    vocab_size = resolve_vocab_size(config)
    header_meta = _header_meta(export_dir)
    scaled_stems = scaled_module_stems(header_meta)
    quant_targets = quantization_targets(export_dir)
    integer_tally = _IntegerTensorTally(vocab_size)

    class_totals = _new_class_totals()
    observed_expert_counts: dict[str, int] = {}
    saw_routed = False
    for key, nbytes in sorted(spans.items()):
        dtype, shape = header_meta.get(key, (None, None))
        klass = classify_read_class(
            _live(key), profile=profile, checkpoint_key=key,
            in_assignment=False, embedding_streamed=embedding.streamed,
            dtype=dtype, shape=shape, vocab_size=vocab_size,
            scaled_stems=scaled_stems, quant_targets=quant_targets,
            context=context, projection=projection)
        integer_tally.note(dtype, shape, nbytes, klass)
        # An exported artifact has no allocator/floor distinction on disk, so
        # every always-active tensor lands in `held_fixed`; the recipe-side
        # dense/held_fixed split is only available pre-export.
        saw_routed |= klass == "routed_experts"
        class_totals[klass]["stored_bytes"] += int(nbytes)
        class_totals[klass]["n_tensors"] += 1

    for name, (_dtype, shape) in header_meta.items():
        if len(shape) == 3 and _has_experts_segment(strip_weight_leaf(name)):
            observed_expert_counts[name] = shape[0]

    ledger_total = sum(
        int(entry["stored_bytes"]) for entry in class_totals.values())
    measured_total = fp.source_checkpoint_bytes(export_dir)[0]
    if ledger_total != measured_total:
        raise ReadTrafficError(
            f"[read_traffic] {context}: classified {ledger_total} of "
            f"{measured_total} tensor-data bytes under {export_dir!r}. Every "
            "shipped byte must be classified; an unclassified remainder is a "
            "silently omitted term in the bandwidth figure."
        )

    # Resident, never per-token traffic -- and outside the shard ledger, so it
    # is added only after the reconciliation above has passed.
    codebook_bytes, codebook_source = _exported_codebook_sidecar_bytes(
        export_dir, context=context)
    if codebook_bytes:
        class_totals["resident_codebooks"]["stored_bytes"] += int(codebook_bytes)
        class_totals["resident_codebooks"]["n_tensors"] += 1

    routing = resolve_routing_factor(
        config, observed_expert_counts=observed_expert_counts, context=context,
    ) if saw_routed else None
    _apply_probabilities(class_totals, routing)
    return _finalize(
        class_totals, routing,
        reconciliation={
            "ledger_stored_bytes": ledger_total,
            "checkpoint_tensor_data_bytes": measured_total,
            "agrees": True,
            "codebook_sidecar_bytes_outside_shards": int(codebook_bytes),
            "codebook_sidecar_source": codebook_source,
        },
        embedding=embedding,
        integer_tally=integer_tally,
        # Deliberately NOT f"...under {export_dir}": during a staged export
        # this function runs against the ATOMIC STAGING directory, which the
        # publishing rename then destroys -- so an embedded absolute path is
        # provenance naming a directory that no longer exists by the time
        # anyone reads the card. The card already records the final location
        # in `model_dir`. Describe WHAT was measured, matching the sibling
        # claim at :1165 ("allocator assignment + source checkpoint spans").
        measured_from="exported safetensors headers",
    )


# ---------------------------------------------------------------------------
# The stamped form
# ---------------------------------------------------------------------------

def _claim_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "value": report["read_gb_per_token"],
        "units": "GB per generated token (decimal GB)",
        "source": report["measured_from"],
        "scope": report["scope"],
        "breakdown": report["breakdown"],
        "excluded": report["excluded"],
        "routing": report["routing"],
        "embedding": report["embedding"],
        "indexed_lookups": report["indexed_lookups"],
        "note": (
            "expected weight bytes streamed per decode token at batch 1 -- "
            "the quantity that sets decode throughput, which a disk-byte "
            "budget cannot see on a sparse MoE"
        ),
    }


def assignment_read_traffic_claim(
    assignment: Mapping[str, str],
    stats: Mapping[str, dict],
    *,
    model_path: str | os.PathLike,
    **kwargs: Any,
) -> dict[str, Any]:
    """:func:`assignment_read_traffic` in the stamped, advisory shape.

    The recipe-side twin of :func:`read_traffic_claim`, for the stages that
    report a bpp *before* an export exists.  Advisory for the same reason: a
    selection run must not die because a bandwidth diagnostic could not be
    computed.
    """
    try:
        report = assignment_read_traffic(
            assignment, stats, model_path=model_path, **kwargs)
    except Exception as exc:
        return {
            "value": None,
            "source": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return _claim_from_report(report)


def read_traffic_claim(
    export_dir: str | os.PathLike | None,
    *,
    profile=None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The value-plus-provenance dict stamped beside ``achieved_bpp``.

    Advisory by construction, exactly like
    :func:`shipcard.allocator_achieved_bpp`'s cross-check: a bug in this
    accounting must never strand a finished export at card-writing time, so
    a failure is reported as a named ``reason`` rather than raised.  The
    number is either measured and complete or absent -- never partial, which
    would be the silent-zero failure this module exists to avoid.
    """
    if not export_dir:
        return {"value": None, "source": None, "reason": "no export directory"}
    try:
        report = exported_checkpoint_read_traffic(
            export_dir, profile=profile, config=config)
    except Exception as exc:  # advisory: never block a finished export
        return {
            "value": None,
            "source": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return _claim_from_report(report)


__all__ = [
    "SCHEMA",
    "READ_SCOPE",
    "READ_CLASS_TABLE",
    "STREAMED_CLASSES",
    "EXCLUDED_CLASSES",
    "ReadTrafficError",
    "RoutingFactor",
    "assignment_read_traffic",
    "assignment_read_traffic_claim",
    "exported_checkpoint_read_traffic",
    "read_traffic_claim",
    "resolve_routing_factor",
    "classify_read_class",
    "read_model_config",
]
