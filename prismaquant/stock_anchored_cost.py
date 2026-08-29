"""Stock-vLLM platform mapping plugin for the anchored cost lane.

``anchored_cost`` is deliberately format-blind: "The core knows no
quantization format names.  A platform mapping plugin owns the candidate
ladder, the shape-transfer equivalence partition, the exact production
renderer hook, the anchor-rung policy, and extra provenance."  Every existing
plugin is CB-family (``cb_anchored_cost``).  This one maps the **stock
compressed-tensors lane** — the formats vanilla vLLM serves natively — so a
model whose menu is ``{NVFP4, FP8_SOURCE}`` can be priced through the anchored
core instead of the streamed ``aura_cost`` path.

WHY A SEPARATE PLUGIN, AND WHY IT IS SHAPED THIS WAY
---------------------------------------------------
The CB lane has a *ladder*: many codebook rungs per unit, one rendered anchor,
and the rest priced by a fitted within-segment shape ratio.  The stock lane
has no ladder at all.  On a native-FP8 checkpoint the whole menu is

  * ``NVFP4`` — the one rung that costs something, and
  * a **source terminal** (``FP8_SOURCE`` / ``BF16``) that is lossless by
    construction because the exporter copies the source slice verbatim.

With exactly one costed rung there is nothing to extrapolate *to*.  This
module therefore refuses the extrapolation machinery outright rather than
degenerating it (§ ``assert_single_rung_partition``).  That refusal is
structural, not stylistic: ``anchored_cost._fit_currency`` raises on any panel
unit with fewer than two rungs, and its design rank must equal the feature
width, so a single-rung segment can never produce a ``ShapeFit`` at all.  The
only way to reach ``price_anchored_candidates`` here would be to hand-build a
``ShapeFit`` claiming panel provenance (arm / payload / panel-receipt digests)
that no panel ever produced — the exact forgery the three-stamp predicates in
``allocator_candidates`` exist to refuse.  So we do not.

THE COST_SOURCE THIS MODULE EMITS, AND WHY IT DIFFERS FROM THE CB LANE
---------------------------------------------------------------------
``allocator_candidates.cost_entry_is_anchored_aura_supersurrogate`` admits an
anchored row on three stamps, all required:
``cost_currency == ANCHORED_AURA_COST_CURRENCY``,
``cost_source == ANCHORED_AURA_COST_SOURCE`` (``"production_arm_render"``) and
an integral ``fisher_application_count == 1``.

``anchored_cost.PricedCell.allocation_entry`` hardcodes
``cost_source="anchored_aura_extrapolation"`` instead, so **no producer in the
live tree emits the blessed stamp** — the CB lane writes
``anchored_aura_extrapolation`` and ``dsv4_aura_cb_reprice`` writes ``"aura"``.
Both fall through to the generic weight-only branch, which happens to read
``predicted_dloss`` directly and so prices identically *today*; what they lose
is the named branch and the scoped
``cost_entry_prices_unmeasured_activation_at_zero`` bypass.  (Archaeology:
``anchored_aura_extrapolation`` landed in 55a464e; the admission constant
landed later in 1db9f41 and the producers were never moved onto it.)

This module emits the blessed stamp, for a reason that is stronger here than
it would be in the CB lane: with one costed rung the priced number **is** the
production-arm render.  ``shape_ratio`` is identically 1.0 — not a fitted
ratio rounded to one, but the absence of a transfer.  Calling that row an
"extrapolation" would be false.  ``production_arm_render`` is the literally
true label, and it is the one the allocator's admission branch reads.

THE SOURCE TERMINAL IS *NOT* SPELLED ``source_passthrough``
----------------------------------------------------------
``cost_entry_is_source_passthrough`` requires membership in
``SOURCE_PASSTHROUGH_FORMATS``, which is derived from
``zero_cost_by_construction`` and today contains only
``{FP8_BLOCK_UE8M0_SOURCE, MXFP4_SOURCE}``.  ``FP8_SOURCE`` and ``BF16`` are
both declared ``zero_cost_by_construction=False`` with the explicit reason
"the cost pipeline emits real rows for it, so its candidate is not
synthesized".  Writing ``cost_source="source_passthrough"`` onto an
``FP8_SOURCE`` row would therefore be a near-miss string the predicate
refuses — two spellings of one contract, which is exactly the failure mode
``anchored_cost``'s re-export comment warns about.

The spelling the production cost pipeline already uses for these two formats
is a plain measured-zero row: ``measure_quant_cost._emit_weight_only_rows``
does ``_accumulate_result(accum, name, spec.name, 0.0, 0.0, 0.0)`` for
``spec.name in ("BF16", "FP8_SOURCE")``, and ``aura_cost._ZERO_COST_FORMATS``
zeroes their ``predicted_dloss``.  ``exact_terminal_cost_entry`` reproduces
that spelling and self-checks that it lands in ``cost_entry_is_bit_exact``.

ACTIVATION-QUANTIZATION BLINDNESS (the named limitation carried forward)
-----------------------------------------------------------------------
``NVFP4`` has ``act_quant_changes_input=True`` while AURA's ``dW`` is
weights-only, so the costed rung's price is activation-quantization-blind —
the same standing limitation ``cost_entry_is_anchored_aura_supersurrogate``
documents for the CB menus, and the same arbiter applies: a served A/B, not a
producer-side assertion.  Unlike the CB case the exposure here is narrower in
one way and wider in another, and both belong on the artifact card:

  * narrower — the menu has one activation-quantizing rung, so the blindness
    cannot reorder rungs *within* a family at all;
  * wider — the only competing rung is an identity-activation terminal, so
    the blindness sits directly on the NVFP4-vs-source margin, which is the
    single margin this lane's whole allocation turns on.

``aqua_activation_cost`` (the AQUA A-side) is the wired remedy and composes
additively through ``cost_entry_act_dloss``; this module leaves ``act_dloss``
absent rather than writing a zero, because "unmeasured" and "zero" are
different claims and the former must stay visible.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    ANCHORED_AURA_COST_CURRENCY,
    ANCHORED_AURA_COST_SOURCE,
    PASSTHROUGH_SOURCE_REQUIREMENTS,
    check_format_applicability,
    cost_entry_is_anchored_aura_supersurrogate,
    cost_entry_is_bit_exact,
)
from prismaquant.anchored_cost import (
    AURA_CURRENCY,
    AnchoredCostError,
    AnchorScalar,
    CandidateSpec,
    PluginDeclaration,
    PricedCell,
    RenderRequest,
    ScalarRenderResult,
    SegmentKey,
    UnitSpec,
    candidates_by_segment,
    make_production_render_receipt_from_hashes,
)
from prismaquant.cost_stage_checkpoint import canonical_json, canonical_json_sha256


class StockAnchoredCostError(AnchoredCostError):
    """A stock-lane ladder, legality, partition, or pricing refusal."""


STOCK_ANCHORED_PLUGIN_SCHEMA = "prismaquant.stock_anchored_plugin.v1"
STOCK_ANCHORED_PAYLOAD_SCHEMA = "prismaquant.stock_anchored_cost_payload.v1"

PLUGIN_ID = "prismaquant.stock_vllm"
PLUGIN_VERSION = "1"

#: The partition this plugin declares to the core.  Stated as an impossibility
#: rather than a policy: there is exactly one costed rung per segment, so no
#: pair of rungs exists between which a ratio could be transferred.
EQUIVALENCE_CONTRACT = (
    "single costed rung per serving unit; the equivalence class is the unit's "
    "own anchor rung, so no within-segment shape transfer exists and none may "
    "be constructed"
)

#: The one costed rung of the stock lane on a native-FP8 or bf16 checkpoint.
DEFAULT_COSTED_FORMAT = "NVFP4"

#: The trivial equivalence class.  One name for every costed candidate: a
#: segment with a single member cannot transfer, and naming it after the
#: format would invite a future reader to add a second member to "the same"
#: class.
SINGLE_RUNG_EQUIVALENCE_CLASS = "single_rung"

#: Terminals live outside the transfer partition entirely (``UnitSpec``
#: refuses ``segment_for`` on a terminal), so these two names are provenance
#: labels, never fit keys.
SOURCE_TERMINAL_FAMILY = "source_terminal"
SOURCE_TERMINAL_BASIS = "source_terminal"

#: Shape features exist only because ``CandidateSpec`` requires a non-empty
#: basis on a non-terminal.  A single all-zero column is deliberate belt and
#: braces: if some future caller reached ``fit_segment_shape`` anyway, the
#: centered design matrix would be identically zero, its rank 0 != 1, and
#: ``_fit_currency`` would refuse before any ratio could be produced.  The
#: per-unit ``len(rows) < 2`` refusal fires first; this is the second lock.
SINGLE_RUNG_SHAPE_FEATURES: tuple[float, ...] = (0.0,)

#: The production render levers, identical to every anchored CB lane
#: (``dsv4_aura_cb_reprice``, ``dense_anchored_cb``, ``rtx4090_fp8_burn``) and
#: to ``build_production_cache``'s ``--enable`` default
#: ``gptq,static_act_order,joint_scale_opt``.  ``weighted_vq`` is inert on
#: non-codebook families (``render_production_weight`` only consults it for
#: ``WEIGHTED_RENDER_FAMILIES``), and is carried so one dict describes the
#: production arm on every lane.
RENDER_LEVERS: Mapping[str, object] = {
    "gptq": True,
    "static_act_order": True,
    "joint_scale_opt": True,
    "weighted_vq": True,
}

#: The named, carried-forward limitation stamped into plugin provenance and
#: into the payload, so an artifact built from this table can state it.
ACTIVATION_BLINDNESS_LIMITATION = (
    "AURA predicted_dloss is a weights-only projection, and the costed rung "
    "NVFP4 quantizes activations (act_quant_changes_input=True). This table "
    "is therefore activation-quantization-blind on the one margin it "
    "allocates over -- costed rung vs identity-activation source terminal. "
    "The A side is not zero; it is unmeasured, and act_dloss is left absent "
    "rather than written as 0.0. AQUA (aqua_activation_cost) is the wired "
    "remedy; a served A/B is the arbiter."
)


# --------------------------------------------------------------------------
# Unit declarations
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StockUnitDeclaration:
    """One serving unit's already source-gated stock-lane ladder.

    The caller owns source legality and byte accounting exactly as in
    ``cb_anchored_cost.CBUnitDeclaration``; this module never re-derives
    either.  ``source_kind`` is carried so the legality gate can be re-run as
    a *check* (``check_declaration_legality``) rather than trusted.
    """

    qname: str
    role: str
    unit_class: str
    n_params: int
    source_kind: str
    costed_format: str
    terminal_format: str
    payload_bytes_by_format: Mapping[str, int]
    serving_group: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "qname", "role", "unit_class", "source_kind",
            "costed_format", "terminal_format",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise StockAnchoredCostError(
                    f"stock unit declaration {name} is empty"
                )
            object.__setattr__(self, name, value)
        if int(self.n_params) <= 0:
            raise StockAnchoredCostError(
                f"{self.qname}: n_params must be positive"
            )
        object.__setattr__(self, "n_params", int(self.n_params))
        costed = fr.canonical_format_name(self.costed_format)
        terminal = fr.canonical_format_name(self.terminal_format)
        if costed == terminal:
            raise StockAnchoredCostError(
                f"{self.qname}: costed rung and terminal are the same format "
                f"{costed!r}; a unit with nothing to trade is not allocatable"
            )
        object.__setattr__(self, "costed_format", costed)
        object.__setattr__(self, "terminal_format", terminal)
        payload = {
            fr.canonical_format_name(str(name)): int(value)
            for name, value in dict(self.payload_bytes_by_format).items()
        }
        for name in (costed, terminal):
            if payload.get(name, 0) <= 0:
                raise StockAnchoredCostError(
                    f"{self.qname}: no positive payload_bytes for {name!r}; "
                    "the allocator's byte accounting is caller-owned and must "
                    "be supplied, never inferred here"
                )
        object.__setattr__(self, "payload_bytes_by_format", payload)


@dataclass(frozen=True)
class LadderRefusal:
    """One unit the stock ladder could not build, with the gate's own reason.

    A refusal is a *finding*, not an error to be smoothed over: a serving unit
    with no legal terminal is the platform reporting a serving gap, which is
    the signal principle 1 preserves.  It is recorded, surfaced by inventory,
    and fails the build closed -- never silently dropped and never handed an
    invented terminal.
    """

    qname: str
    format_name: str
    kind: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "qname": self.qname,
            "format": self.format_name,
            "kind": self.kind,
            "reason": self.reason,
            "detail": self.detail,
        }


def check_declaration_legality(
    declaration: StockUnitDeclaration,
    *,
    shape: Sequence[int],
    target_profile: str,
) -> tuple[LadderRefusal, ...]:
    """Re-run the real legality gate over one declared ladder.

    Deliberately a re-check rather than a construction step.  The caller
    already gated its ladder; running ``check_format_applicability`` again
    here means a stock-lane table can never contain a rung the serving profile
    or the passthrough-integrity rule would refuse, and -- more useful -- it
    names *which* gate refused, in the gate's own vocabulary, so the blocker
    that reaches a human is the platform's sentence and not this module's
    paraphrase.
    """
    refusals: list[LadderRefusal] = []
    for kind, format_name in (
        ("costed_rung", declaration.costed_format),
        ("source_terminal", declaration.terminal_format),
    ):
        verdict = check_format_applicability(
            tuple(int(value) for value in shape),
            format_name,
            qname=declaration.qname,
            source_kind=declaration.source_kind,
            target_profile=target_profile,
        )
        if not verdict.legal:
            refusals.append(LadderRefusal(
                declaration.qname,
                format_name,
                kind,
                str(verdict.reason or "illegal"),
                str(verdict.detail or ""),
            ))
    if not refusals:
        terminal = declaration.terminal_format
        if terminal not in PASSTHROUGH_SOURCE_REQUIREMENTS:
            refusals.append(LadderRefusal(
                declaration.qname, terminal, "source_terminal",
                "not_a_passthrough_format",
                f"{terminal} has no source-passthrough contract; a terminal "
                "must be lossless by construction, not merely cheap",
            ))
        elif fr.get_format(terminal).act_quant_changes_input:
            refusals.append(LadderRefusal(
                declaration.qname, terminal, "source_terminal",
                "terminal_quantizes_activations",
                f"{terminal} changes the activation path, so shipping the "
                "weights verbatim does not make the unit free",
            ))
    return tuple(refusals)


# --------------------------------------------------------------------------
# The plugin
# --------------------------------------------------------------------------
class StockAnchoredFormatPlugin:
    """The five declarations ``anchored_cost.AnchoredFormatPlugin`` requires.

    ``renderer`` is the production-arm hook.  It is ``None`` by default and
    ``render`` then fail-closes: there is no RTN fallback and no render-free
    path, because RTN-vs-production ``dW`` is result-changing (decisive at
    fp8) and a cost measured on a render the exporter will not ship is a
    rendering confound, not a cheaper measurement.
    """

    def __init__(
        self,
        *,
        arm_identity: Mapping[str, object],
        serving_profile_id: str,
        costed_format: str = DEFAULT_COSTED_FORMAT,
        renderer: Callable[
            [RenderRequest], ScalarRenderResult | Mapping[str, object]
        ] | None = None,
    ) -> None:
        arm = dict(canonical_json(
            arm_identity, where="stock anchored production arm identity",
        ))
        if not arm:
            raise StockAnchoredCostError(
                "stock anchored production arm identity is empty"
            )
        profile_id = str(serving_profile_id).strip()
        if not profile_id:
            raise StockAnchoredCostError("serving profile id is empty")
        self._arm_identity = arm
        self._serving_profile_id = profile_id
        self._costed_format = fr.canonical_format_name(str(costed_format))
        self._renderer = renderer

    # -- read-only views ---------------------------------------------------
    @property
    def arm_identity(self) -> Mapping[str, object]:
        return dict(self._arm_identity)

    @property
    def costed_format(self) -> str:
        return self._costed_format

    @property
    def serving_profile_id(self) -> str:
        return self._serving_profile_id

    # -- Protocol ----------------------------------------------------------
    def plugin_identity(self) -> PluginDeclaration:
        return PluginDeclaration(
            plugin_id=PLUGIN_ID,
            plugin_version=PLUGIN_VERSION,
            equivalence_contract=EQUIVALENCE_CONTRACT,
        )

    def describe_candidate(
        self, unit: UnitSpec, format_name: str,
    ) -> CandidateSpec:
        """Return the unit's own declared candidate, unchanged.

        ``candidates_by_segment`` requires ``resolved == declared``; returning
        the identical object is how the CB plugin satisfies that too.  The
        checks below are re-derivations, so a ladder built by some other code
        path cannot smuggle in a candidate whose family / class disagree with
        this plugin's partition.
        """
        canonical = fr.canonical_format_name(str(format_name))
        try:
            candidate = next(
                item for item in unit.candidates
                if item.format_name == canonical
            )
        except StopIteration as exc:
            raise StockAnchoredCostError(
                f"{unit.qname}: plugin was asked for undeclared {canonical!r}"
            ) from exc
        if candidate.terminal:
            if candidate.family != SOURCE_TERMINAL_FAMILY:
                raise StockAnchoredCostError(
                    f"{unit.qname}/{canonical}: terminal family differs"
                )
            return candidate
        if candidate.format_name != self._costed_format:
            raise StockAnchoredCostError(
                f"{unit.qname}: {canonical!r} is a second costed rung; this "
                "plugin declares exactly one, because a second rung would "
                "make a shape transfer expressible"
            )
        expected_family = fr.get_format(canonical).family
        if candidate.family != expected_family:
            raise StockAnchoredCostError(
                f"{unit.qname}/{canonical}: family {candidate.family!r} "
                f"differs from the registry's {expected_family!r}"
            )
        if candidate.equivalence_class != SINGLE_RUNG_EQUIVALENCE_CLASS:
            raise StockAnchoredCostError(
                f"{unit.qname}/{canonical}: equivalence class differs from "
                f"the declared {SINGLE_RUNG_EQUIVALENCE_CLASS!r}"
            )
        return candidate

    def select_anchor(
        self,
        unit: UnitSpec,
        segment: SegmentKey,
        candidates: Sequence[CandidateSpec],
    ) -> str:
        """The anchor policy is forced, not chosen.

        A segment here holds exactly one candidate, so "which rung do we
        render" has one answer.  The refusal below is the anchor-policy half
        of the no-extrapolation guarantee: the moment a segment held two
        rungs, this plugin would stop rather than silently anchor one and
        transfer to the other.
        """
        if len(candidates) != 1:
            raise StockAnchoredCostError(
                f"{unit.qname}/{segment.stamp}: {len(candidates)} costed "
                "rungs in one segment; the stock lane declares exactly one "
                "and cannot transfer between rungs"
            )
        only = candidates[0]
        if only.terminal or only.format_name != self._costed_format:
            raise StockAnchoredCostError(
                f"{unit.qname}/{segment.stamp}: segment member "
                f"{only.format_name!r} is not the declared costed rung "
                f"{self._costed_format!r}"
            )
        return only.format_name

    def render(
        self, request: RenderRequest,
    ) -> ScalarRenderResult | Mapping[str, object]:
        if self._renderer is None:
            raise StockAnchoredCostError(
                "stock scalar renderer is not installed; refusing an RTN or "
                "render-free fallback -- the anchor must be the production "
                "render the exporter ships"
            )
        if request.purpose != "anchor":
            raise StockAnchoredCostError(
                f"{request.qname}: purpose {request.purpose!r} is a panel or "
                "validation render, which only a fitted lane needs; the stock "
                "lane renders anchors only"
            )
        return self._renderer(request)

    def provenance_identity_fields(self) -> Mapping[str, object]:
        return {
            "schema": STOCK_ANCHORED_PLUGIN_SCHEMA,
            "arm_identity": self._arm_identity,
            "segment_key_fields": ["family", "role", "equivalence_class"],
            "equivalence_vocabulary_name": SINGLE_RUNG_EQUIVALENCE_CLASS,
            "costed_format": self._costed_format,
            "serving_profile_id": self._serving_profile_id,
            "render_levers": dict(RENDER_LEVERS),
            "aura_is_only_cost_currency": True,
            "extrapolation_expressible": False,
            "activation_blindness_limitation": ACTIVATION_BLINDNESS_LIMITATION,
        }


# --------------------------------------------------------------------------
# Ladder construction
# --------------------------------------------------------------------------
def build_stock_units(
    declarations: Sequence[StockUnitDeclaration],
    plugin: StockAnchoredFormatPlugin,
) -> tuple[UnitSpec, ...]:
    """Convert caller-owned ladders into core ``UnitSpec``s.

    Each unit gets exactly two candidates: the costed rung and the source
    terminal.  ``UnitSpec`` independently enforces "exactly one terminal" and
    "at least one renderable", so a declaration that lost its terminal to the
    legality gate fails here rather than reaching the allocator with a
    silently one-sided menu.
    """
    units: list[UnitSpec] = []
    seen: set[str] = set()
    for declaration in sorted(declarations, key=lambda item: item.qname):
        if declaration.qname in seen:
            raise StockAnchoredCostError(
                f"duplicate stock unit {declaration.qname!r}"
            )
        seen.add(declaration.qname)
        if declaration.costed_format != plugin.costed_format:
            raise StockAnchoredCostError(
                f"{declaration.qname}: declared costed rung "
                f"{declaration.costed_format!r} differs from the plugin's "
                f"{plugin.costed_format!r}"
            )
        payload = declaration.payload_bytes_by_format
        costed_bytes = int(payload[declaration.costed_format])
        terminal_bytes = int(payload[declaration.terminal_format])
        if terminal_bytes <= costed_bytes:
            raise StockAnchoredCostError(
                f"{declaration.qname}: terminal {declaration.terminal_format} "
                f"({terminal_bytes} B) is no larger than the costed rung "
                f"({costed_bytes} B); a free terminal that is also no bigger "
                "dominates the menu and the unit is not a decision"
            )
        n_params = int(declaration.n_params)
        costed = CandidateSpec(
            format_name=declaration.costed_format,
            bits=8.0 * costed_bytes / n_params,
            payload_bytes=costed_bytes,
            family=fr.get_format(declaration.costed_format).family,
            equivalence_class=SINGLE_RUNG_EQUIVALENCE_CLASS,
            shape_features=SINGLE_RUNG_SHAPE_FEATURES,
            coordinate=8.0 * costed_bytes / n_params,
        )
        terminal = CandidateSpec(
            format_name=declaration.terminal_format,
            bits=8.0 * terminal_bytes / n_params,
            payload_bytes=terminal_bytes,
            family=SOURCE_TERMINAL_FAMILY,
            equivalence_class=SOURCE_TERMINAL_BASIS,
            shape_features=(),
            coordinate=0.0,
            terminal=True,
            # The terminal is a real allocator choice here (that is the whole
            # point of the menu), and its activation path is the identity --
            # re-checked, not assumed, because a terminal that quantized
            # activations would be priced at zero for a cost it does incur.
            allocator_selectable=not fr.get_format(
                declaration.terminal_format
            ).act_quant_changes_input,
        )
        units.append(UnitSpec(
            qname=declaration.qname,
            role=declaration.role,
            unit_class=declaration.unit_class,
            candidates=(costed, terminal),
            n_params=n_params,
            serving_group=declaration.serving_group,
        ))
    return tuple(units)


def assert_single_rung_partition(
    units: Sequence[UnitSpec],
    plugin: StockAnchoredFormatPlugin,
) -> None:
    """Prove no extrapolation is expressible over these units.

    Three things are checked, and together they close every route by which a
    price could be transferred from one rung to another:

      1. each unit resolves to exactly one segment, so no unit spans a
         partition boundary;
      2. that segment holds exactly one costed candidate, so ``ShapeFit.ratio``
         has no distinct (target, anchor) pair to be called with;
      3. that candidate is the plugin's declared costed rung, so a foreign
         rung cannot be smuggled in as "the anchor" of a segment it does not
         belong to.

    ``candidates_by_segment`` is the core's own grouping, so this asserts on
    the same object the core would price from -- not on a parallel view of it.
    """
    for unit in sorted(units, key=lambda item: item.qname):
        grouped = candidates_by_segment(unit, plugin)
        if len(grouped) != 1:
            raise StockAnchoredCostError(
                f"{unit.qname}: {len(grouped)} transfer segments; the stock "
                "lane declares exactly one so that no transfer is expressible"
            )
        (segment, candidates), = grouped.items()
        if len(candidates) != 1:
            raise StockAnchoredCostError(
                f"{unit.qname}/{segment.stamp}: {len(candidates)} costed "
                "rungs; two rungs in one segment would make a shape ratio "
                "expressible, which this lane forbids by construction"
            )
        if candidates[0].format_name != plugin.costed_format:
            raise StockAnchoredCostError(
                f"{unit.qname}/{segment.stamp}: costed rung "
                f"{candidates[0].format_name!r} is not the plugin's "
                f"{plugin.costed_format!r}"
            )


# --------------------------------------------------------------------------
# Anchors from an already-measured scalar table
# --------------------------------------------------------------------------
#: What a measured anchor row must say about how its ``dW`` was produced.
#: ``aura_cost``'s ``_delta_w`` emits ``"rendered"``/``"rtn"``; the streamed
#: production-anchor path emits ``"production_render"`` and adds
#: ``production_anchor_measured``.  Only the latter is admissible: RTN-vs-
#: production ``dW`` is result-changing, so an anchor rendered by anything but
#: the production arm is a rendering confound wearing the right field name.
PRODUCTION_RENDER_DW_SOURCE = "production_render"


def anchors_from_measured_scalars(
    requests: Sequence[RenderRequest],
    measured: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    arm_identity: Mapping[str, object],
    payload_identity: Mapping[str, object],
) -> dict[tuple[str, SegmentKey], AnchorScalar]:
    """Wrap already-measured production scalars as core ``AnchorScalar``s.

    The stock analogue of ``cb_anchored_cost.anchors_from_streamed_payload``,
    and the interface this lane actually mirrors: the CB plugin computes no
    tensors either -- it consumes finished rows from the AURA runner and
    re-wraps them.  This exists because AURA's ``gW`` is a *live autograd*
    quantity, harvested from ``weight.grad`` inside a post-accumulate hook and
    nulled immediately; there is no on-disk adjoint a per-unit ``render()``
    could re-derive, and re-running the KL-adjoint backward once per unit
    would multiply the probe cost by the unit count.

    So the batched harvest stays batched, and this function is the seam
    between it and the anchored core's identity machinery.  Receipts are built
    from once-hashed global identities, exactly as the streamed CB adapters
    do, so per-cell work stays O(1) in the identity size.
    """
    arm_sha = canonical_json_sha256(
        arm_identity, where="stock measured anchor production arm identity",
    )
    payload_sha = canonical_json_sha256(
        payload_identity, where="stock measured anchor payload identity",
    )
    anchors: dict[tuple[str, SegmentKey], AnchorScalar] = {}
    for request in requests:
        if request.purpose != "anchor":
            raise StockAnchoredCostError(
                f"{request.qname}: {request.purpose!r} request cannot become "
                "an anchor"
            )
        key = (request.qname, request.segment)
        if key in anchors:
            raise StockAnchoredCostError(f"duplicate anchor {key}")
        try:
            row = measured[request.qname][request.format_name]
        except (KeyError, TypeError) as exc:
            raise StockAnchoredCostError(
                f"{request.qname}/{request.format_name}: no measured "
                "production anchor"
            ) from exc
        if not isinstance(row, Mapping):
            raise StockAnchoredCostError(
                f"{request.qname}/{request.format_name}: measured row is not "
                "a mapping"
            )
        if row.get("dw_source") != PRODUCTION_RENDER_DW_SOURCE:
            raise StockAnchoredCostError(
                f"{request.qname}/{request.format_name}: dw_source "
                f"{row.get('dw_source')!r} is not {PRODUCTION_RENDER_DW_SOURCE!r}; "
                "refusing an RTN or otherwise non-production anchor"
            )
        if row.get("production_anchor_measured") is not True:
            raise StockAnchoredCostError(
                f"{request.qname}/{request.format_name}: row does not attest "
                "production_anchor_measured"
            )
        scalar = ScalarRenderResult(
            float(row["predicted_dloss"]),
            (
                float(row["weight_mse_diagnostic"])
                if row.get("weight_mse_diagnostic") is not None else None
            ),
        )
        receipt = make_production_render_receipt_from_hashes(
            request,
            scalar,
            arm_identity_sha256=arm_sha,
            payload_identity_sha256=payload_sha,
        )
        anchors[key] = AnchorScalar(
            request.qname,
            request.segment,
            request.format_name,
            scalar.predicted_dloss,
            receipt,
        )
    return anchors


# --------------------------------------------------------------------------
# Cost rows
# --------------------------------------------------------------------------
def exact_terminal_cost_entry(format_name: str) -> dict[str, object]:
    """The zero row for a lossless source terminal.

    Spelled exactly as the production cost pipeline spells it -- see the
    module docstring for why ``cost_source="source_passthrough"`` is *wrong*
    for ``FP8_SOURCE`` and ``BF16``.  The final assertion is the point of the
    function: it proves the row lands in the branch this module intends
    (``cost_entry_is_bit_exact`` -> ``cost_entry_is_exact_by_construction``
    -> a hard 0.0 that outranks any noisy surrogate), rather than leaving that
    to a reader's belief about precedence.
    """
    canonical = fr.canonical_format_name(str(format_name))
    if canonical not in PASSTHROUGH_SOURCE_REQUIREMENTS:
        raise StockAnchoredCostError(
            f"{canonical} has no source-passthrough contract; refusing to "
            "price it at zero"
        )
    if fr.get_format(canonical).act_quant_changes_input:
        raise StockAnchoredCostError(
            f"{canonical} quantizes activations; a verbatim weight copy does "
            "not make it free and this row would price a real cost at zero"
        )
    entry: dict[str, object] = {
        "predicted_dloss": 0.0,
        "weight_mse": 0.0,
        "output_mse": 0.0,
        "output_mse_measured": False,
    }
    if not cost_entry_is_bit_exact(entry, canonical):
        raise StockAnchoredCostError(
            f"{canonical}: terminal row does not satisfy the allocator's "
            "exact-by-construction predicate; refusing to emit a row whose "
            "pricing branch is not the intended one"
        )
    return entry


def price_single_rung_candidates(
    units: Sequence[UnitSpec],
    plugin: StockAnchoredFormatPlugin,
    anchors: Mapping[tuple[str, SegmentKey], AnchorScalar],
) -> dict[str, dict[str, dict[str, object]]]:
    """Price every unit from its own production-arm render. No transfer.

    The deliberate non-use of ``anchored_cost.price_anchored_candidates`` is
    the module's central design decision; the docstring at the top explains
    why fabricating the ``ShapeFit`` it requires would be a provenance
    forgery.  What is *kept* from that function is every check that still has
    meaning without a fit: same-segment anchor, request identity, and the
    production-arm digest binding.
    """
    assert_single_rung_partition(units, plugin)
    arm_sha = canonical_json_sha256(
        plugin.arm_identity, where="stock anchored pricing production arm",
    )
    rows: dict[str, dict[str, dict[str, object]]] = {}
    for unit in sorted(units, key=lambda item: item.qname):
        (segment, candidates), = candidates_by_segment(unit, plugin).items()
        candidate = candidates[0]
        anchor = anchors.get((unit.qname, segment))
        if anchor is None:
            raise StockAnchoredCostError(
                f"{unit.qname}/{segment.stamp}: no same-segment real anchor"
            )
        expected_request = RenderRequest(
            unit.qname, segment, candidate.format_name, "anchor",
        )
        if (
            anchor.qname != unit.qname
            or anchor.segment != segment
            or anchor.format_name != candidate.format_name
            or anchor.receipt.request != expected_request
        ):
            raise StockAnchoredCostError(
                f"{unit.qname}/{segment.stamp}: anchor identity differs from "
                "the unit it is supposed to price"
            )
        if anchor.receipt.arm_identity_sha256 != arm_sha:
            raise StockAnchoredCostError(
                f"{unit.qname}: anchor was rendered by a different production "
                "arm than this plugin declares"
            )
        value = float(anchor.predicted_dloss)
        if not math.isfinite(value) or value < 0.0:
            raise StockAnchoredCostError(
                f"{unit.qname}: anchored AURA is not finite and nonnegative"
            )
        cell = PricedCell(
            qname=unit.qname,
            candidate=candidate,
            predicted_dloss=value,
            segment=segment,
            # The blessed admission stamp. With one rung the priced number IS
            # the production-arm render, so this is the literally true label
            # and not a relabelled extrapolation (module docstring).
            cost_source=ANCHORED_AURA_COST_SOURCE,
            anchor_format=candidate.format_name,
            anchor_predicted_dloss=value,
            # Identically one: the absence of a transfer, not a fitted ratio
            # that happened to round to one.
            shape_ratio=1.0,
            anchor_receipt_sha256=anchor.receipt.receipt_sha256,
            arm_identity_sha256=arm_sha,
        )
        entry = cell.allocation_entry()
        terminal = next(
            item for item in unit.candidates if item.terminal
        )
        terminal_entry = exact_terminal_cost_entry(terminal.format_name)
        terminal_entry["memory_bytes"] = int(terminal.payload_bytes)
        rows[unit.qname] = {
            candidate.format_name: entry,
            terminal.format_name: terminal_entry,
        }
    assert_stock_cost_table(rows, plugin)
    return rows


def assert_stock_cost_table(
    rows: Mapping[str, Mapping[str, Mapping[str, object]]],
    plugin: StockAnchoredFormatPlugin,
) -> None:
    """Fail closed unless every emitted row lands in its intended consumer.

    The stock-lane analogue of ``anchored_cost.assert_aura_only_cost_table``,
    and the reason this module does not simply call that one: the core's
    version requires ``cost_source == "anchored_aura_extrapolation"``, which
    is precisely the label this lane must not use.  Everything else it checks
    is reproduced -- forbidden parallel cost inputs, the h^2 guard, and the
    presence of a real render receipt -- plus two checks only this lane can
    make: that ``shape_ratio`` is exactly 1.0, and that each row actually
    satisfies the allocator predicate it is written for.
    """
    forbidden = frozenset({
        "h_trace", "cw_m2", "imatrix_dispersion", "activation_mse",
    })
    for qname, per_format in sorted(rows.items()):
        anchored = [
            (name, entry) for name, entry in per_format.items()
            if entry.get("cost_source") == ANCHORED_AURA_COST_SOURCE
        ]
        if len(anchored) != 1:
            raise StockAnchoredCostError(
                f"{qname}: {len(anchored)} anchored rows; the stock lane "
                "prices exactly one costed rung per unit"
            )
        name, entry = anchored[0]
        if name != plugin.costed_format:
            raise StockAnchoredCostError(
                f"{qname}: anchored row is {name!r}, not the declared costed "
                f"rung {plugin.costed_format!r}"
            )
        present = sorted(forbidden & set(entry))
        if present:
            raise StockAnchoredCostError(
                f"{qname}/{name}: forbidden parallel cost inputs {present}; "
                "AURA predicted_dloss already carries the Fisher"
            )
        if entry.get("cost_currency") != ANCHORED_AURA_COST_CURRENCY:
            raise StockAnchoredCostError(
                f"{qname}/{name}: cost currency is not the AURA projection"
            )
        if entry.get("shape_ratio") != 1.0:
            raise StockAnchoredCostError(
                f"{qname}/{name}: shape_ratio {entry.get('shape_ratio')!r} is "
                "not exactly 1.0; a transferred price cannot exist in a lane "
                "with one costed rung"
            )
        if not entry.get("anchor_receipt_sha256") or not entry.get(
            "arm_identity_sha256"
        ):
            raise StockAnchoredCostError(
                f"{qname}/{name}: anchored row has no production render "
                "receipt"
            )
        if not cost_entry_is_anchored_aura_supersurrogate(dict(entry)):
            raise StockAnchoredCostError(
                f"{qname}/{name}: row fails the allocator's anchored-AURA "
                "admission; it would be priced through the generic branch "
                "instead of the branch it was written for"
            )
        terminals = [
            (other, row) for other, row in per_format.items()
            if other != name
        ]
        if len(terminals) != 1:
            raise StockAnchoredCostError(
                f"{qname}: {len(terminals)} terminal rows; expected exactly "
                "one source terminal"
            )
        terminal_name, terminal_entry = terminals[0]
        if not cost_entry_is_bit_exact(dict(terminal_entry), terminal_name):
            raise StockAnchoredCostError(
                f"{qname}/{terminal_name}: terminal row is not "
                "exact-by-construction to the allocator"
            )


# --------------------------------------------------------------------------
# Pinned (profile-denied) units
# --------------------------------------------------------------------------
def pinned_passthrough_rows(
    pinned: Mapping[str, tuple[str, int]],
) -> dict[str, dict[str, dict[str, object]]]:
    """Rows for units the serving profile pins to their source precision.

    These never become ``UnitSpec``s: ``UnitSpec.__post_init__`` demands a
    renderable candidate and a pinned unit has none by definition, so the core
    would refuse them.  They still need rows, because a cost table that simply
    omits a serving unit is indistinguishable from one that forgot it.

    ``pinned`` maps qname -> (terminal_format, payload_bytes).
    """
    rows: dict[str, dict[str, dict[str, object]]] = {}
    for qname, (format_name, payload_bytes) in sorted(pinned.items()):
        if int(payload_bytes) <= 0:
            raise StockAnchoredCostError(
                f"{qname}: pinned payload_bytes must be positive"
            )
        entry = exact_terminal_cost_entry(format_name)
        entry["memory_bytes"] = int(payload_bytes)
        rows[qname] = {fr.canonical_format_name(str(format_name)): entry}
    return rows


# --------------------------------------------------------------------------
# Packed-expert empirical merge
# --------------------------------------------------------------------------
#: What ``expert_empirical_cost`` measures is a serving-unit KL, not an AURA
#: projection: the smooth cost is route-flip-blind for routed experts, so a
#: rendered-dW surrogate cannot see the cost that matters there.  Its rows
#: therefore keep their own provenance verbatim.  Re-stamping them
#: ``production_arm_render`` would be the forgery this module refuses in the
#: other direction, and would also claim a currency they are not in.
EMPIRICAL_EXPERT_COST_SOURCE_FIELD = "cost_source"


@dataclass(frozen=True)
class MergeReport:
    """What the disjoint union did, in numbers a gate can read."""

    anchored_units: int
    pinned_units: int
    empirical_units: int
    total_units: int
    total_cells: int
    empirical_formats: tuple[str, ...] = ()
    missing_empirical: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "anchored_units": self.anchored_units,
            "pinned_units": self.pinned_units,
            "empirical_units": self.empirical_units,
            "total_units": self.total_units,
            "total_cells": self.total_cells,
            "empirical_formats": list(self.empirical_formats),
            "missing_empirical": list(self.missing_empirical),
        }


def expert_empirical_rows(
    payload: Mapping[str, object],
) -> dict[str, dict[str, dict[str, object]]]:
    """Read the standalone ``expert_empirical_cost`` payload's cost rows.

    Deliberately reads the *standalone* artifact -- the one written with no
    ``--merge-base`` -- rather than calling
    ``expert_empirical_cost.merge_cost_payloads``.  That function is the
    shipping merge for ``run-pipeline.sh``'s two lanes and folds exactly two
    provenances; this lane has three (anchored, empirical, pinned) and needs
    the pinned third to be checked for overlap too.  Its rows are copied
    verbatim: the empirical measurement is a serving-unit KL in its own
    currency, and re-stamping it into the anchored currency would claim a
    projection that was never computed.
    """
    if not isinstance(payload, Mapping):
        raise StockAnchoredCostError(
            "expert empirical payload is not a mapping"
        )
    schema = str(payload.get("schema") or "")
    if schema != "prismaquant.expert_empirical_cost.v1":
        raise StockAnchoredCostError(
            f"expert empirical payload schema {schema!r} is not "
            "prismaquant.expert_empirical_cost.v1; refusing to merge an "
            "artifact whose contract is unknown"
        )
    costs = payload.get("costs")
    if not isinstance(costs, Mapping) or not costs:
        raise StockAnchoredCostError(
            "expert empirical payload carries no cost rows"
        )
    rows: dict[str, dict[str, dict[str, object]]] = {}
    for qname, per_format in costs.items():
        if not isinstance(per_format, Mapping) or not per_format:
            raise StockAnchoredCostError(
                f"{qname}: expert empirical row is empty"
            )
        converted: dict[str, dict[str, object]] = {}
        for name, entry in per_format.items():
            if not isinstance(entry, Mapping):
                raise StockAnchoredCostError(
                    f"{qname}/{name}: expert empirical cell is not a mapping"
                )
            if "predicted_dloss" not in entry:
                raise StockAnchoredCostError(
                    f"{qname}/{name}: expert empirical cell has no "
                    "predicted_dloss; the allocator would fall back to "
                    "h_trace x weight_mse on a row whose h_trace is a "
                    "deliberate 0.0"
                )
            converted[str(name)] = dict(entry)
        rows[str(qname)] = converted
    return rows


def merge_cost_rows(
    *,
    anchored: Mapping[str, Mapping[str, Mapping[str, object]]],
    pinned: Mapping[str, Mapping[str, Mapping[str, object]]],
    empirical: Mapping[str, Mapping[str, Mapping[str, object]]] | None,
    expected_empirical_units: Sequence[str] = (),
    require_empirical: bool = True,
) -> tuple[dict[str, dict[str, dict[str, object]]], MergeReport]:
    """Disjoint union of the three provenances into one cost table.

    Every serving unit must be priced by exactly one of

      * an anchored production-arm render (dense-ish quantizable units),
      * a measured empirical serving-unit KL (packed routed experts), or
      * an exact source terminal (units the serving profile pins),

    and an overlap is a bug, not a preference: two provenances for one unit
    means one of them is being silently discarded, and which one would depend
    on dict ordering.  So overlap fails closed rather than resolving by
    precedence.

    ``require_empirical`` exists because the packed-expert measurement is a
    separate, long GPU job.  When its output is absent the merge does not
    quietly ship a table missing 84 units -- it names them.
    """
    anchored_keys = set(anchored)
    pinned_keys = set(pinned)
    empirical_keys = set(empirical or {})
    for left_name, left, right_name, right in (
        ("anchored", anchored_keys, "pinned", pinned_keys),
        ("anchored", anchored_keys, "empirical", empirical_keys),
        ("pinned", pinned_keys, "empirical", empirical_keys),
    ):
        overlap = sorted(left & right)
        if overlap:
            raise StockAnchoredCostError(
                f"{len(overlap)} unit(s) priced by both {left_name} and "
                f"{right_name} provenance, e.g. {overlap[:5]}; a serving unit "
                "has exactly one price and an overlap silently discards one"
            )
    expected = tuple(dict.fromkeys(str(name) for name in expected_empirical_units))
    missing = tuple(name for name in expected if name not in empirical_keys)
    if missing and require_empirical:
        raise StockAnchoredCostError(
            f"{len(missing)} packed-expert unit(s) have no empirical "
            f"serving-unit KL, e.g. {list(missing[:5])}; refusing to emit a "
            "cost table that silently omits them. Run expert_empirical_cost "
            "first, or pass require_empirical=False to produce an explicitly "
            "incomplete inventory table."
        )
    merged: dict[str, dict[str, dict[str, object]]] = {}
    for source in (anchored, empirical or {}, pinned):
        for qname, per_format in source.items():
            merged[str(qname)] = {
                str(name): dict(entry) for name, entry in per_format.items()
            }
    empirical_formats = tuple(sorted({
        str(name)
        for per_format in (empirical or {}).values()
        for name in per_format
    }))
    report = MergeReport(
        anchored_units=len(anchored_keys),
        pinned_units=len(pinned_keys),
        empirical_units=len(empirical_keys),
        total_units=len(merged),
        total_cells=sum(len(row) for row in merged.values()),
        empirical_formats=empirical_formats,
        missing_empirical=missing,
    )
    return merged, report


def assert_probe_coverage(
    costs: Mapping[str, Mapping[str, Mapping[str, object]]],
    probe_stats: Mapping[str, object],
) -> tuple[str, ...]:
    """Refuse a cost table the allocator would silently half-read.

    ``allocator.py`` loads ``stats`` from ``--probe`` and ``costs`` from
    ``--costs``; the candidate loop then iterates over **stats** and does
    ``if name not in costs: continue`` (``allocator_candidates.py:1811``).
    A cost row whose qname is missing from the probe is therefore never turned
    into a candidate, never allocated, and -- the part that costs a night --
    never reported.  A byte budget computed from a table with silently absent
    units undershoots invisibly.

    Returns the probe units this table does not price, so a caller can decide
    whether that set is the expected pinned remainder or a bug.  Raises only
    for the unambiguous direction: a priced unit the probe never saw.
    """
    priced = set(costs)
    known = set(probe_stats)
    unknown = sorted(priced - known)
    if unknown:
        raise StockAnchoredCostError(
            f"{len(unknown)} priced unit(s) are absent from probe['stats'], "
            f"e.g. {unknown[:5]}; the allocator iterates stats and would "
            "silently drop every one of them"
        )
    return tuple(sorted(known - priced))


def build_stock_allocator_cost_payload(
    *,
    costs: Mapping[str, Mapping[str, Mapping[str, object]]],
    merge_report: MergeReport,
    plugin: StockAnchoredFormatPlugin,
    campaign_identity: Mapping[str, object],
    unpriced_probe_units: Sequence[str] = (),
    refusals: Sequence[LadderRefusal] = (),
    extra_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The allocator-facing payload, with its provenance attached.

    Five top-level keys, mirroring ``build_cb_allocator_cost_payload``.
    There is deliberately **no** ``"stats"`` key: the allocator reads stats
    from ``--probe`` and ignores any it finds here, so shipping a second copy
    would create a silent authority conflict between two files.

    ``formats`` is present because ``allocator.py`` reads
    ``cost_data["formats"]`` unconditionally.  ``cost_mode`` is stamped
    because ``run-pipeline.sh`` refuses to reuse a table produced under a
    different estimator, and an unstamped table is reused "unverified" --
    which is how a run silently allocates on the wrong cost.
    ``ladder_refusals`` travels *inside* the payload rather than beside it, so
    a serving gap cannot be lost in the gap between a log and an artifact.
    """
    costs_out = {
        str(qname): {
            str(name): dict(entry) for name, entry in per_format.items()
        }
        for qname, per_format in sorted(costs.items())
    }
    formats = sorted({name for row in costs_out.values() for name in row})
    return {
        "schema": STOCK_ANCHORED_PAYLOAD_SCHEMA,
        "formats": formats,
        "costs": costs_out,
        "provenance": {
            "schema": STOCK_ANCHORED_PAYLOAD_SCHEMA,
            "cost_mode": "aura",
            "cost_currency": AURA_CURRENCY,
            "fisher_application_count": 1,
            "plugin_identity": plugin.plugin_identity().to_dict(),
            "plugin_provenance": dict(plugin.provenance_identity_fields()),
            "campaign_identity": dict(canonical_json(
                campaign_identity, where="stock anchored campaign identity",
            )),
            "merge_report": merge_report.to_dict(),
            "ladder_refusals": [item.to_dict() for item in refusals],
            "unpriced_probe_units": sorted(str(x) for x in unpriced_probe_units),
            "extrapolation_expressible": False,
            "activation_blindness_limitation": ACTIVATION_BLINDNESS_LIMITATION,
            **dict(extra_provenance or {}),
        },
        "meta": {
            "cost_currency": AURA_CURRENCY,
            "cost_semantics": (
                "one production-arm render per serving unit, read directly; "
                "shape_ratio is identically 1.0 because a single costed rung "
                "admits no transfer. No h_trace, imatrix dispersion, "
                "weight-MSE or parallel activation cost is applied."
            ),
            "unit_count": len(costs_out),
            "cell_count": sum(len(row) for row in costs_out.values()),
        },
    }
