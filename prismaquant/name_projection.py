"""One shared projection between PrismaQuant's parameter-name namespaces.

A Linear in this pipeline has four names, and until this module every
consumer re-derived the mapping between them ad hoc — which is how a
weight consumed through a skipped module class (DSv4's ``attn.wo_a``)
could stay invisible to the allocator while its bytes shipped. This
module is the ONE projection layer: after it lands, no consumer keeps a
private name mapping.

The namespaces (all names are LOGICAL — whole-tensor, device- and
rank-independent; see "Tensor Parallelism" below):

* ``live``      — loaded-transformers parameter names, exactly what
  :attr:`prismaquant.model_walk.WalkNode.name` carries
  (``model.layers.0.self_attn.q_proj.weight``). Tensor spelling: the
  parameter leaf is part of the name.
* ``recipe``    — allocator/probe/stats/assignment qnames
  (``model.layers.0.self_attn.q_proj``, module spelling without the
  leaf). This is the namespace probe stats rows, format plans,
  assignment dicts and cost tables key on.
* ``checkpoint``— SOURCE safetensors tensor keys (the floor's byte
  authority). Tensor spelling.
* ``export``    — exported-artifact safetensors keys written by an
  export lane. Tensor spelling. May intentionally differ from
  ``checkpoint`` (packed-expert splitting, multimodal conventions).
* ``vllm``      — vLLM-internal module qnames that scheme dispatch
  compares against at load time. Module spelling.

Every mapping routes through :class:`prismaquant.model_profiles.base.ModelProfile`
accessors — the profile owns the knowledge, this module owns the
discipline:

* **Fail closed and loud.** A name with no mapping raises
  :class:`NameProjectionError` carrying structured fields (name,
  source and target namespace, a machine-readable ``code``, and what
  was tried). No silent ``None``, no best-effort guess.
* **Declared drops are data, not failures.**
  ``checkpoint_to_live_name`` returns ``None`` BY CONTRACT for keys
  outside the live graph (visual towers, MTP sidecars, FP8 scale
  siblings). That outcome is surfaced as
  :data:`DECLARED_OUT_OF_GRAPH` on a :class:`ProjectedName` so callers
  branch on a field, never on prose.
* **Non-totality lives in the type.** Where one logical tensor spans
  several spellings or several tensors collapse into one serving unit,
  the API says so: reverse lookups return/require unique members via
  explicit errors, packed-expert aggregates expose their member lists
  as tuples, and fused sibling groups report kind + members on
  :class:`ServingGroup`. Nothing collapses silently.

Round-trip property (pinned by tests): where the profile's rules are
prefix rewrites — the normal case — ``recipe -> checkpoint -> recipe``
is the identity, and ``recipe_to_live`` inverts ``live_to_recipe``
exactly over the supplied universe. Where the rules are NOT invertible
(DSv4's flat checkpoint naming has no ``source_tensor_name`` override,
so the live→checkpoint direction is declared identity while the real
mapping is not invertible from a single name), the layer reports the
declared answer and the asymmetry stays visible instead of being papered
over with a guessed inverse.

Tensor Parallelism
------------------
Names here are properties of the LOGICAL tensor. No method accepts or
returns a rank, shard index, or TP degree, and node identity never
changes with TP degree — a per-rank spelling is a SEPARATE decoration
over a logical name, deliberately absent until the serving lane
publishes a machine-readable shard contract. Byte accounting does not
live here at all: totals stay on
:class:`prismaquant.model_walk.WalkNode` and per-device readings go
through :func:`prismaquant.model_walk.per_device_bytes`.

This module imports only the standard library at module scope; torch
and model_walk types enter lazily through :meth:`NameProjection.for_walk`.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

__all__ = [
    "CHECKPOINT",
    "DECLARED_OUT_OF_GRAPH",
    "EXPORT",
    "GROUP_FUSED_SIBLINGS",
    "GROUP_PACKED_EXPERTS",
    "GROUP_SINGLETON",
    "LIVE",
    "MAPPED",
    "NAMESPACES",
    "NameProjection",
    "NameProjectionError",
    "packed_expert_alias",
    "PROJECTABLE_PAIRS",
    "ProjectedName",
    "RECIPE",
    "ServingGroup",
    "VLLM",
    "strip_weight_leaf",
]

# ---------------------------------------------------------------------------
# Namespaces and outcomes
# ---------------------------------------------------------------------------

#: Loaded-transformers parameter names (what WalkNode.name carries).
LIVE = "live"
#: Allocator/probe/stats/assignment qnames (module spelling).
RECIPE = "recipe"
#: Source safetensors tensor keys.
CHECKPOINT = "checkpoint"
#: Exported-artifact safetensors tensor keys.
EXPORT = "export"
#: vLLM-internal scheme-dispatch module qnames.
VLLM = "vllm"

NAMESPACES = (LIVE, RECIPE, CHECKPOINT, EXPORT, VLLM)

#: Outcome of a projection that the profile DECLINES by contract (a key
#: outside the live graph). Data for callers to branch on — never an
#: exception, never a silent None.
DECLARED_OUT_OF_GRAPH = "declared_out_of_graph"
#: Outcome of a successful projection.
MAPPED = "mapped"

_PROJECTABLE_PAIRS = {
    (LIVE, RECIPE),
    (RECIPE, LIVE),
    (RECIPE, VLLM),
    (CHECKPOINT, LIVE),
    (LIVE, CHECKPOINT),
    (RECIPE, EXPORT),
}
#: The (source, target) pairs :meth:`NameProjection.project` serves.
PROJECTABLE_PAIRS = tuple(sorted(_PROJECTABLE_PAIRS))


def strip_weight_leaf(tensor_name: str) -> str:
    """Drop a trailing ``.weight``; leave any other name unchanged.

    The recipe namespace spells units WITHOUT the parameter leaf (stats
    rows, assignment dicts, hook targets), while live/checkpoint/export
    spellings carry it. This module owns that conversion so consumers
    stop re-deriving it with private ``_recipe_name`` helpers.
    """
    name = str(tensor_name)
    return name[: -len(".weight")] if name.endswith(".weight") else name


def _default_expert_parent_for_projection(projection_name: str) -> str | None:
    """No-profile fallback for the per-expert -> packed projection mapping.

    Mirrors ``ModelProfile.packed_expert_parent_for_projection``'s legacy
    fallback: per-expert ``gate_proj``/``up_proj`` fuse into the packed
    ``gate_up_proj`` (output-axis cat, the transformers packed-FusedMoE
    convention); ``down_proj`` packs 1:1. Anything else (e.g. MiniMax's
    per-expert ``w1``/``w2``/``w3`` modules, which stay per-expert live)
    has no packed parent here — callers with a profile should pass its
    ``packed_expert_parent_for_projection`` instead.
    """
    if projection_name in ("gate_proj", "up_proj"):
        return "gate_up_proj"
    if projection_name == "down_proj":
        return "down_proj"
    return None


def packed_expert_alias(qname: str, parent_for_projection=None) -> str | None:
    """Packed live qname a per-expert Linear aggregates into, or None.

    ``...experts.{i}.{proj}`` -> ``...experts.{parent}`` when
    ``parent_for_projection(proj)`` names a packed parent
    (``ModelProfile.packed_expert_parent_for_projection``; the legacy
    gate/up/down fallback when None). Non-expert names and unrecognized
    projections return None.

    Lives HERE because it is name derivation, not byte accounting: it is
    the one structural ``experts.{idx}.{leaf}`` parser, shared by
    :meth:`NameProjection.packed_parent_of_expert_param` and by
    ``footprint.source_tensor_bytes_manifest``'s dual-entry convention
    (a per-expert span is covered under BOTH spellings), and it mirrors
    the profile layer's own packed-format grouping detection
    (``packed_expert_format_group``).
    """
    parts = str(qname).split(".")
    if len(parts) < 3 or parts[-3] != "experts" or not parts[-2].isdigit():
        return None
    fn = (parent_for_projection if parent_for_projection is not None
          else _default_expert_parent_for_projection)
    parent = fn(parts[-1])
    if not parent:
        return None
    return ".".join(parts[:-2] + [str(parent)])


@dataclasses.dataclass(frozen=True)
class ProjectedName:
    """One resolved projection. Callers branch on :attr:`outcome`."""

    source: str
    source_namespace: str
    target: str                  # None iff outcome == DECLARED_OUT_OF_GRAPH
    target_namespace: str
    outcome: str                 # MAPPED | DECLARED_OUT_OF_GRAPH
    via: str                     # the profile accessor that produced it


@dataclasses.dataclass(frozen=True)
class ServingGroup:
    """The profile-declared serving group of one unit.

    ``kind`` distinguishes the three shapes a unit can take; ``key`` is
    the group identity in the queried namespace and is ``""`` only when
    ``kind == GROUP_SINGLETON``. For grouped kinds, ``members`` lists
    the member names PRESENT IN THE SUPPLIED LIVE UNIVERSE (empty when
    no universe was given — use :meth:`NameProjection.require_serving_group`
    when absence would be a defect); for singletons it is the unit
    itself.
    """

    kind: str                    # GROUP_SINGLETON | GROUP_FUSED_SIBLINGS | GROUP_PACKED_EXPERTS
    key: str
    namespace: str               # namespace of `key` and `members` (mirrors the query)
    members: tuple[str, ...]


GROUP_SINGLETON = "singleton"
GROUP_FUSED_SIBLINGS = "fused_siblings"
GROUP_PACKED_EXPERTS = "packed_experts"


class NameProjectionError(KeyError):
    """A projection failed closed.

    Every field is structured so callers branch on fields, never on
    message text: ``code`` discriminates the failure class
    (``unknown_namespace``, ``unmapped_in_universe``, ``ambiguous``,
    ``no_universe_supplied``, ``malformed_profile_result``,
    ``profile_accessor_failed``, ``unsupported_pair``); ``tried`` names
    each mechanism that was attempted; the message exists for humans.
    """

    def __init__(
        self,
        *,
        name: str,
        source_namespace: str,
        target_namespace: str,
        code: str,
        tried: Iterable[str] = (),
        detail: str = "",
    ) -> None:
        self.name = str(name)
        self.source_namespace = str(source_namespace)
        self.target_namespace = str(target_namespace)
        self.code = str(code)
        self.tried = tuple(str(t) for t in tried)
        self.detail = str(detail)
        message = (
            f"name_projection: cannot map {self.name!r} from "
            f"{self.source_namespace!r} to {self.target_namespace!r} "
            f"[{self.code}]"
        )
        if self.detail:
            message += f": {self.detail}"
        if self.tried:
            message += "; tried: " + "; ".join(self.tried)
        super().__init__(message)

    def __str__(self) -> str:  # KeyError.__str__ reprs its argument
        return self.args[0]


# ---------------------------------------------------------------------------
# The projection
# ---------------------------------------------------------------------------


class NameProjection:
    """All name projections for one model family, derived from one profile.

    Build once per process and pass to every consumer::

        proj = NameProjection.for_walk(load_walk(path).result, profile)
        stats_key = proj.recipe_unit(node.name)          # probe/cost join
        live = proj.checkpoint_to_live(ckpt_key)         # footprint/read-traffic join

    ``live_names`` seeds the universe the reverse directions need: the
    walk supplies it naturally (:meth:`for_walk`), and a consumer may
    supply any comparable enumeration instead. Reverse queries without a
    universe raise ``code="no_universe_supplied"`` rather than guessing;
    forward queries never need one. ``checkpoint_keys`` (a source span
    scan, e.g. ``footprint.source_tensor_span_bytes``) additionally
    enables the coverage queries (:meth:`checkpoint_keys_for`,
    :meth:`require_checkpoint_span`) whose empty answer is the loud gate
    the footprint migration needs.
    """

    def __init__(
        self,
        profile,
        *,
        live_names: Sequence[str] = (),
        checkpoint_keys: Sequence[str] | None = None,
    ) -> None:
        if profile is None:
            raise ValueError(
                "NameProjection requires a ModelProfile; a None profile "
                "would silently degrade every mapping to identity")
        self._profile = profile
        self._live_names = tuple(dict.fromkeys(str(n) for n in live_names))
        # Forward index once, O(universe): recipe spellings of every live
        # name, keyed both ways. Tied weights keep distinct recipes, so a
        # collision here means the PROFILE collapses distinct tensors and
        # the reverse direction is genuinely ambiguous — recorded, then
        # raised at query time with both candidates named.
        self._recipe_param_by_live: dict[str, str] = {}
        self._live_by_recipe_param: dict[str, list[str]] = {}
        self._live_by_unit: dict[str, list[str]] = {}
        for live in self._live_names:
            recipe_param = self.live_to_recipe(live)
            self._recipe_param_by_live[live] = recipe_param
            self._live_by_recipe_param.setdefault(recipe_param, []).append(live)
            self._live_by_unit.setdefault(
                strip_weight_leaf(recipe_param), []).append(live)
        self._ckpt_supplied = checkpoint_keys is not None
        self._ckpt_keys_by_target = self._build_checkpoint_index(
            checkpoint_keys or ())
        # Group membership among the universe, keyed by canonical group
        # key (module spelling): the present-in-universe view that makes
        # many->one groups inspectable without a second enumeration.
        self._members_by_group: dict[tuple[str, str], list[str]] = {}
        for live in self._live_names:
            group = self.serving_group(live)
            if group.kind != GROUP_SINGLETON:
                self._members_by_group.setdefault(
                    (group.namespace, group.key), []).append(live)

    @classmethod
    def for_walk(
        cls,
        result,
        profile,
        *,
        checkpoint_keys: Sequence[str] | None = None,
    ) -> "NameProjection":
        """Build the projection over a walked model's node universe."""
        from .model_walk import WalkResult  # noqa: F401  (type guard only)

        if not isinstance(result, WalkResult):
            raise TypeError(
                "for_walk expects a prismaquant.model_walk.WalkResult, "
                f"got {type(result).__name__}")
        return cls(
            profile,
            live_names=[node.name for node in result.nodes],
            checkpoint_keys=checkpoint_keys,
        )

    # ------------------------------------------------------------ basics --

    @property
    def profile(self):
        return self._profile

    def live_names(self) -> tuple[str, ...]:
        """The live universe backing the reverse directions."""
        return self._live_names

    # ---------------------------------------------------- forward paths --
    # Each wrapper validates the accessor's RESULT type and wraps accessors
    # failures; none swallows exceptions into a fallback (that best-effort
    # shape is decision_units.fused_group_key's, and it is precisely what
    # this layer must not repeat).

    def live_to_recipe(self, live_name: str) -> str:
        """Live parameter/module name -> recipe spelling (leaf preserved).

        Total over the live graph: profiles declare it identity unless a
        multimodal staging splits the namespaces.
        """
        return self._map_module_form(
            live_name, LIVE, RECIPE, "live_to_recipe_name")

    def recipe_unit(self, live_name: str) -> str:
        """Live name -> the recipe UNIT key (module spelling).

        This is THE join for probe stats rows, format plans, assignment
        dicts, cost tables, and hook targets. It is ``strip_weight_leaf``
        composed with :meth:`live_to_recipe`, owned here so no consumer
        keeps a private ``_recipe_name``.
        """
        return strip_weight_leaf(self.live_to_recipe(live_name))

    def to_vllm_internal(self, name: str) -> str:
        """Recipe/checkpoint spelling -> vLLM scheme-dispatch qname.

        Accepts module or parameter spelling; the leaf survives (the
        exact-match rules some specs declare operate on module qnames,
        so the accessor is applied to the module form and the leaf
        reattached — the same composition
        ``structure.build_model_graph`` uses).
        """
        module_form = strip_weight_leaf(name)
        vllm_module = self._call_accessor(
            "to_vllm_internal_name", module_form, source=RECIPE, target=VLLM)
        leaf = name[len(module_form):] if name != module_form else ""
        if leaf and not vllm_module.endswith(leaf):
            return f"{vllm_module}{leaf}"
        return vllm_module

    def checkpoint_to_live(
        self, ckpt_key: str, *, multimodal: bool = False,
    ) -> ProjectedName:
        """Source checkpoint key -> live name, or the DECLARED drop.

        Never raises for a key the profile declines by contract (visual
        towers, MTP sidecars, scale siblings): that is
        :data:`DECLARED_OUT_OF_GRAPH`, the field read-traffic classifies
        ``excluded_non_text_graph`` from. Raises only when the accessor
        itself misbehaves.
        """
        try:
            mapped = self._profile.checkpoint_to_live_name(
                str(ckpt_key), multimodal=multimodal)
        except Exception as exc:
            raise NameProjectionError(
                name=str(ckpt_key), source_namespace=CHECKPOINT,
                target_namespace=LIVE, code="profile_accessor_failed",
                tried=("ModelProfile.checkpoint_to_live_name",),
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        if mapped is None:
            return ProjectedName(
                source=str(ckpt_key), source_namespace=CHECKPOINT,
                target=None, target_namespace=LIVE,
                outcome=DECLARED_OUT_OF_GRAPH,
                via="ModelProfile.checkpoint_to_live_name",
            )
        if not isinstance(mapped, str) or not mapped:
            raise NameProjectionError(
                name=str(ckpt_key), source_namespace=CHECKPOINT,
                target_namespace=LIVE, code="malformed_profile_result",
                tried=("ModelProfile.checkpoint_to_live_name",),
                detail=f"expected a non-empty str or None, got {mapped!r}",
            )
        # The accessor returns a PARAMETER spelling; normalize a trailing
        # .weight off exactly as footprint's manifest does, so the target
        # matches allocator unit qnames.
        return ProjectedName(
            source=str(ckpt_key), source_namespace=CHECKPOINT,
            target=strip_weight_leaf(mapped), target_namespace=LIVE,
            outcome=MAPPED, via="ModelProfile.checkpoint_to_live_name",
        )

    def live_to_source(self, live_param: str) -> str:
        """Live parameter name -> the profile's declared SOURCE spelling.

        HONEST LIMIT: ``source_tensor_name`` documents export/source
        naming and is NOT guaranteed to invert
        ``checkpoint_to_live_name`` (DSv4 overrides the latter with its
        flat naming but leaves the former identity — its docstring's
        inverse claim has no code behind it). Treat this as the DECLARED
        spelling; the authoritative live->checkpoint direction is the
        universe-backed :meth:`checkpoint_keys_for` once a span scan is
        supplied.
        """
        try:
            mapped = self._profile.source_tensor_name(str(live_param))
        except Exception as exc:
            raise NameProjectionError(
                name=str(live_param), source_namespace=LIVE,
                target_namespace=CHECKPOINT, code="profile_accessor_failed",
                tried=("ModelProfile.source_tensor_name",),
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        if not isinstance(mapped, str) or not mapped:
            raise NameProjectionError(
                name=str(live_param), source_namespace=LIVE,
                target_namespace=CHECKPOINT, code="malformed_profile_result",
                tried=("ModelProfile.source_tensor_name",),
                detail=f"expected a non-empty str, got {mapped!r}",
            )
        return mapped

    def recipe_to_export(self, recipe_param: str) -> str:
        """Recipe parameter name -> exported-artifact tensor key."""
        try:
            mapped = self._profile.export_tensor_name(str(recipe_param))
        except Exception as exc:
            raise NameProjectionError(
                name=str(recipe_param), source_namespace=RECIPE,
                target_namespace=EXPORT, code="profile_accessor_failed",
                tried=("ModelProfile.export_tensor_name",),
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        if not isinstance(mapped, str) or not mapped:
            raise NameProjectionError(
                name=str(recipe_param), source_namespace=RECIPE,
                target_namespace=EXPORT, code="malformed_profile_result",
                tried=("ModelProfile.export_tensor_name",),
                detail=f"expected a non-empty str, got {mapped!r}",
            )
        return mapped

    # ----------------------------------------------------- reverse paths --
    # Index-backed over the live universe; they refuse loudly rather than
    # guess when the universe is missing or the inverse is ambiguous.

    def _unique_inverse(
        self, recipe_name: str, tables: Sequence[tuple[dict[str, list[str]], str]],
    ) -> str:
        wanted = str(recipe_name)
        tried: list[str] = []
        found: list[str] = []
        for table, label in tables:
            tried.append(f"{label} over the live universe "
                         f"({len(self._live_names)} name(s))")
            candidates = table.get(wanted)
            if candidates:
                found = candidates
                break
        if not found:
            raise NameProjectionError(
                name=wanted, source_namespace=RECIPE,
                target_namespace=LIVE, code="unmapped_in_universe",
                tried=tuple(tried),
                detail="no live name maps to this recipe spelling under "
                       "this profile; a fused-group or packed-aggregate "
                       "KEY is not itself a live tensor — ask for a member",
            )
        if len(found) > 1:
            raise NameProjectionError(
                name=wanted, source_namespace=RECIPE,
                target_namespace=LIVE, code="ambiguous",
                tried=tuple(tried),
                detail=f"{len(found)} live names map to this recipe "
                       f"spelling ({', '.join(sorted(found))}); the "
                       "inverse is not unique, resolve via serving_group()",
            )
        return found[0]

    def recipe_to_live(self, recipe_name: str) -> str:
        """Unique live parameter behind a recipe spelling (inverse of
        :meth:`live_to_recipe`). Accepts unit or parameter spelling."""
        return self._unique_inverse(recipe_name, (
            (self._live_by_recipe_param, "live_to_recipe"),
            (self._live_by_unit, "recipe_unit"),
        ))

    def unit_to_live(self, unit_qname: str) -> str:
        """Unique live parameter behind a recipe UNIT key (module spelling)."""
        return self._unique_inverse(unit_qname, (
            (self._live_by_unit, "recipe_unit"),
        ))

    # --------------------------------------------------- coverage joins --

    def _build_checkpoint_index(self, checkpoint_keys: Sequence[str]) -> dict[str, tuple[str, ...]]:
        """Map live/recipe/packed-aggregate targets -> covering ckpt keys.

        Mirrors footprint's dual-entry manifest convention: a per-expert
        checkpoint key covers BOTH its own name and its packed-parent
        aggregate, so a coverage query succeeds under either naming
        scheme and an uncovered unit is genuinely uncovered.
        """
        covered: dict[str, set[str]] = {}

        def _cover(target: str, key: str) -> None:
            covered.setdefault(strip_weight_leaf(target), set()).add(key)

        for raw in checkpoint_keys:
            key = str(raw)
            projected = self.checkpoint_to_live(key)
            if projected.outcome != MAPPED:
                continue  # declared out-of-graph: covers nothing live
            live_base = projected.target
            _cover(live_base, key)
            packed = self.packed_parent_of_expert_param(live_base)
            if packed is not None:
                _cover(packed, key)
        return {target: tuple(sorted(keys)) for target, keys in covered.items()}

    def checkpoint_keys_for(self, target: str) -> tuple[str, ...]:
        """Checkpoint keys whose bytes cover this live/unit name.

        Empty means genuinely uncovered — combine with
        :meth:`require_checkpoint_span` where emptiness is a refusal.
        """
        if not self._ckpt_supplied:
            raise NameProjectionError(
                name=str(target), source_namespace=LIVE,
                target_namespace=CHECKPOINT, code="no_universe_supplied",
                tried=("checkpoint-key index",),
                detail="constructed without checkpoint_keys; pass a source "
                       "span scan (footprint.source_tensor_span_bytes) to "
                       "enable coverage queries",
            )
        return self._ckpt_keys_by_target.get(strip_weight_leaf(str(target)), ())

    def require_checkpoint_span(self, target: str) -> tuple[str, ...]:
        """:meth:`checkpoint_keys_for`, refusing an uncovered unit.

        This is the coverage assertion footprint.md asked for: a decided
        unit with no source span is an accounting bug, and this call is
        where it becomes loud instead of a silent floor term.
        """
        keys = self.checkpoint_keys_for(target)
        if not keys:
            raise NameProjectionError(
                name=str(target), source_namespace=LIVE,
                target_namespace=CHECKPOINT, code="uncovered_unit",
                tried=(f"checkpoint-key index over "
                       f"{len(self._ckpt_keys_by_target)} covered target(s)",),
                detail="no supplied checkpoint key maps to this unit (or to "
                       "its packed aggregate)",
            )
        return keys

    # --------------------------------------------------------- grouping --

    def fused_sibling_group(self, qname: str) -> str | None:
        """Profile-declared fused-sibling group key, or None.

        Unlike ``decision_units.fused_group_key`` this does NOT swallow
        profile exceptions into an identity fallback: an accessor that
        fails is a declaration bug and refuses here.
        """
        try:
            group = self._profile.fused_sibling_group(strip_weight_leaf(qname))
        except Exception as exc:
            raise NameProjectionError(
                name=str(qname), source_namespace=RECIPE, target_namespace=RECIPE,
                code="profile_accessor_failed",
                tried=("ModelProfile.fused_sibling_group",),
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        if group is None:
            return None
        if not isinstance(group, str) or not group:
            raise NameProjectionError(
                name=str(qname), source_namespace=RECIPE, target_namespace=RECIPE,
                code="malformed_profile_result",
                tried=("ModelProfile.fused_sibling_group",),
                detail=f"expected a non-empty str or None, got {group!r}",
            )
        return group

    def packed_format_group(self, qname: str) -> str | None:
        """Profile-declared packed-expert serving-format group, or None."""
        try:
            group = self._profile.packed_expert_format_group(
                strip_weight_leaf(qname))
        except Exception as exc:
            raise NameProjectionError(
                name=str(qname), source_namespace=RECIPE, target_namespace=RECIPE,
                code="profile_accessor_failed",
                tried=("ModelProfile.packed_expert_format_group",),
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        if group is None:
            return None
        if not isinstance(group, str) or not group:
            raise NameProjectionError(
                name=str(qname), source_namespace=RECIPE, target_namespace=RECIPE,
                code="malformed_profile_result",
                tried=("ModelProfile.packed_expert_format_group",),
                detail=f"expected a non-empty str or None, got {group!r}",
            )
        return group

    def serving_group(self, qname: str) -> ServingGroup:
        """The one serving group of a unit: packed experts first, then
        fused siblings, else singleton.

        Precedence matters where both match (an unpacked expert
        projection's leaf can carry a fused-sibling name too): routed
        experts must group by their STACK, so packed wins. The key
        inherits the queried namespace; members are the group's names
        present in the live universe.
        """
        base = strip_weight_leaf(qname)
        packed = self.packed_format_group(base)
        if packed is not None:
            return ServingGroup(
                kind=GROUP_PACKED_EXPERTS, key=packed, namespace=RECIPE,
                members=tuple(sorted(
                    self._members_by_group.get((RECIPE, packed), ())),
                ),
            )
        fused = self.fused_sibling_group(base)
        if fused is not None:
            return ServingGroup(
                kind=GROUP_FUSED_SIBLINGS, key=fused, namespace=RECIPE,
                members=tuple(sorted(
                    self._members_by_group.get((RECIPE, fused), ()))),
            )
        return ServingGroup(
            kind=GROUP_SINGLETON, key="", namespace=RECIPE,
            members=(base,),
        )

    def require_serving_group(self, qname: str, *, kinds: Sequence[str]) -> ServingGroup:
        """Refuse unless the unit's serving group is of a required kind.

        Cost declarations branch on group kind (CB stacks need packed
        groups; dense fusion needs fused ones). Passing an empty
        ``kinds`` asserts the unit is grouped at all.
        """
        group = self.serving_group(qname)
        allowed = tuple(kinds)
        if not allowed:
            if group.kind == GROUP_SINGLETON:
                raise NameProjectionError(
                    name=str(qname), source_namespace=RECIPE,
                    target_namespace=RECIPE, code="ungrouped_unit",
                    tried=("serving_group",),
                    detail="unit carries no profile-declared group",
                )
            return group
        if group.kind not in allowed:
            raise NameProjectionError(
                name=str(qname), source_namespace=RECIPE,
                target_namespace=RECIPE, code="wrong_group_kind",
                tried=("serving_group",),
                detail=f"group kind is {group.kind!r}, required one of "
                       f"{list(allowed)}",
            )
        return group

    def block_id(self, qname: str) -> str:
        """Architectural block id of a unit (shared helper, not a copy)."""
        from .decision_units import block_id_from_qname

        return block_id_from_qname(strip_weight_leaf(str(qname)))

    # -------------------------------------------------- packed experts --

    def packed_parent_of_expert_param(self, qname: str) -> str | None:
        """Packed aggregate a per-expert parameter folds into, or None.

        None means "not a per-expert parameter" — the same declared
        meaning :func:`packed_expert_alias` gives (defined in this
        module, shared with footprint's manifest), applied with the
        profile's own parent mapping.
        """
        base = strip_weight_leaf(str(qname))
        try:
            return packed_expert_alias(
                base, self._profile.packed_expert_parent_for_projection)
        except Exception as exc:
            raise NameProjectionError(
                name=base, source_namespace=RECIPE, target_namespace=RECIPE,
                code="profile_accessor_failed",
                tried=("packed_expert_alias("
                       "ModelProfile.packed_expert_parent_for_projection)",),
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

    # -------------------------------------------------------- dispatch --

    def project(
        self, name: str, source_namespace: str, target_namespace: str,
        *, multimodal: bool = False,
    ) -> ProjectedName:
        """Namespace-generic entry point; always returns a
        :class:`ProjectedName` (branch on ``outcome``) or raises a
        :class:`NameProjectionError`.

        Serves exactly the pairs with a declared accessor or a built
        index — see :data:`PROJECTABLE_PAIRS`. Composing chains (live ->
        recipe -> vllm) is the CALLER'S decision; this method never
        invents a hop the profile did not declare.
        """
        pair = (str(source_namespace), str(target_namespace))
        for ns in pair:
            if ns not in NAMESPACES:
                raise NameProjectionError(
                    name=str(name), source_namespace=pair[0],
                    target_namespace=pair[1], code="unknown_namespace",
                    tried=(),
                    detail=f"{ns!r} is not one of {list(NAMESPACES)}",
                )
        if pair not in _PROJECTABLE_PAIRS:
            raise NameProjectionError(
                name=str(name), source_namespace=pair[0],
                target_namespace=pair[1], code="unsupported_pair",
                tried=(),
                detail=f"projectable pairs: {PROJECTABLE_PAIRS}",
            )
        source_ns, target_ns = pair
        if (source_ns, target_ns) == (CHECKPOINT, LIVE):
            return self.checkpoint_to_live(name, multimodal=multimodal)
        if (source_ns, target_ns) == (RECIPE, EXPORT):
            return ProjectedName(
                source=str(name), source_namespace=source_ns,
                target=self.recipe_to_export(name), target_namespace=target_ns,
                outcome=MAPPED, via="ModelProfile.export_tensor_name",
            )
        if (source_ns, target_ns) == (LIVE, CHECKPOINT):
            return ProjectedName(
                source=str(name), source_namespace=source_ns,
                target=self.live_to_source(name), target_namespace=target_ns,
                outcome=MAPPED, via="ModelProfile.source_tensor_name",
            )
        if (source_ns, target_ns) == (RECIPE, VLLM):
            return ProjectedName(
                source=str(name), source_namespace=source_ns,
                target=self.to_vllm_internal(name), target_namespace=target_ns,
                outcome=MAPPED, via="ModelProfile.to_vllm_internal_name",
            )
        if (source_ns, target_ns) == (LIVE, RECIPE):
            return ProjectedName(
                source=str(name), source_namespace=source_ns,
                target=self.live_to_recipe(name), target_namespace=target_ns,
                outcome=MAPPED, via="ModelProfile.live_to_recipe_name",
            )
        # (RECIPE, LIVE) — the one index-backed pair.
        return ProjectedName(
            source=str(name), source_namespace=source_ns,
            target=self.recipe_to_live(name), target_namespace=target_ns,
            outcome=MAPPED,
            via="reverse index over the live universe",
        )

    # ------------------------------------------------------- internals --

    def _map_module_form(
        self, name: str, source_ns: str, target_ns: str, accessor: str,
    ) -> str:
        """Apply a MODULE-form profile accessor with leaf preservation.

        Prefix rewrites survive the leaf either way, but specs also
        declare EXACT rules against bare module qnames (qwen3_5's
        ``lm_head``), so the accessor runs on the module form and the
        leaf is reattached — mirroring structure.build_model_graph.
        """
        module_form = strip_weight_leaf(name)
        mapped_module = self._call_accessor(
            accessor, module_form, source=source_ns, target=target_ns)
        leaf = name[len(module_form):] if name != module_form else ""
        if leaf:
            return f"{mapped_module}{leaf}"
        return mapped_module

    def _call_accessor(
        self, accessor: str, arg: str, *, source: str, target: str,
    ) -> str:
        method = getattr(self._profile, accessor, None)
        if not callable(method):
            raise NameProjectionError(
                name=arg, source_namespace=source, target_namespace=target,
                code="profile_accessor_failed", tried=(accessor,),
                detail=f"profile {type(self._profile).__name__} has no "
                       f"callable {accessor}()",
            )
        try:
            mapped = method(arg)
        except Exception as exc:
            raise NameProjectionError(
                name=arg, source_namespace=source, target_namespace=target,
                code="profile_accessor_failed", tried=(accessor,),
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        if not isinstance(mapped, str) or not mapped:
            raise NameProjectionError(
                name=arg, source_namespace=source, target_namespace=target,
                code="malformed_profile_result", tried=(accessor,),
                detail=f"expected a non-empty str, got {mapped!r}",
            )
        return mapped
