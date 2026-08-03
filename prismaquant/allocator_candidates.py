"""Candidate construction and coupled-candidate aggregation."""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import format_registry as fr
from .activation_fair_pricing import (
    APPLIED_MARKER_KEY,
    BRANCH_ACTIVATION_IDENTITY,
    BRANCH_BIT_EXACT,
    BRANCH_CALIBRATED,
    BRANCH_MEASURED,
    BRANCH_SOURCE_PASSTHROUGH,
    BRANCH_UNCALIBRATED,
    ActivationFairPricing,
    CalibrationRow,
)
from .activation_fair_pricing import calibrate as _calibrate_activation_pricing
from .allocator_solver import (
    Candidate,
    PackedExpertRoleUnknown,
    _shape_from_stats,
    predicted_dloss,
)
from .nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_tensor_payload_breakdown,
    is_cb_format,
)
from .serving_profiles import (
    SERVING_LANE_SCHEMA,
    check_serving_format,
    check_serving_shape,
    gridbook_runtime_version,
    serving_lane_route,
)

# The provenance string a source-passthrough candidate carries in place of a
# measured cost. It is NOT an estimator name like ``output_mse`` or
# ``weight_mse``: it says the row was never measured because there is nothing
# to measure.
SOURCE_PASSTHROUGH_COST_SOURCE = "source_passthrough"

# Serving-route token for a passthrough whose bytes the model's OWN loader
# consumes, with no Gridbook codec in the path.
ROUTE_DELEGATED_NATIVE = "delegated_native"

# What a MEASUREMENT says about serving a passthrough's bytes. These are
# verdicts from a real serve attempt on real hardware, not design intent —
# which matters, because on DSv4-Flash/GB10 the measured answers came out the
# OPPOSITE way round from the obvious guess.
ROUTE_STATUS_BACKED = "backed"          # measured serving, possibly with a requirement
ROUTE_STATUS_PENDING = "pending"        # no verdict yet; unaudited
ROUTE_STATUS_BLOCKED = "blocked"        # measured, every known route dead


@dataclass(frozen=True)
class SourcePassthroughContract:
    """One native source format the producer can ship back UNCHANGED.

    "Any format the model natively uses belongs on a passthrough menu": the
    allocator's cheapest honest option for a unit is always to keep the bytes
    the checkpoint already has. This table is that menu, one entry per
    (source format, unit contract) the census finds, so adding a newly
    encountered native format is a data change rather than a new code path.

    Fields:
      ``source_kind``  the token ``_scan_source_dtype_manifest`` stamps on a
        unit whose stored bytes ARE this format. It is the whole legality
        gate: the format is legal exactly where the source already is it, in
        both directions — BF16 is masked on mxfp4 experts and MXFP4_SOURCE is
        masked on the bf16 embedding by the identical rule.
      ``zero_cost_by_construction`` whether the allocator SYNTHESIZES the
        candidate rather than requiring a cost-table column. True for formats
        no cost run will ever have a column for.
      ``serving_route``  the P5b lane route id.
      ``wire_format_id``  the closed-enum spelling the artifact declares and
        the serving side reads (quant_config.json ``source_passthrough``).
        Distinct from ``format_name``: the producer's registry name is ours to
        rename, the wire id is a cross-repo contract.
      ``route_status``  what a MEASUREMENT says about serving these bytes, per
        ``ROUTE_STATUS_*``. Never remove a rung from the menu — an allocator
        that wants an unservable passthrough is reporting a serving gap, and
        hiding the rung would hide the signal — but anything other than
        ``backed`` makes the export fail closed without an explicit override.
      ``route_requirement``  the serving-side condition that makes a BACKED
        route actually fire (e.g. a non-default vLLM MoE backend). Belongs in
        the artifact's serving notes; a backed route with an unmet
        requirement serves no better than a blocked one.
      ``route_evidence``  what was measured, and on what hardware. A route
        verdict without its evidence is a rumour.
      ``detail``  why this entry exists, for the mask/provenance records.
    """

    format_name: str
    source_kind: str
    zero_cost_by_construction: bool
    serving_route: str
    route_status: str
    detail: str
    wire_format_id: str | None = None
    route_requirement: str | None = None
    route_evidence: str | None = None

    @property
    def route_backed(self) -> bool:
        """Whether a serve route for these bytes is known to exist."""
        return self.route_status == ROUTE_STATUS_BACKED


SOURCE_PASSTHROUGH_CONTRACTS: dict[str, SourcePassthroughContract] = {
    contract.format_name: contract
    for contract in (
        # Legacy DeepSeek-V3 / MiniMax block-FP8: FP32 ``weight_scale_inv``.
        # Serves through stock compressed-tensors block-fp8 today.
        SourcePassthroughContract(
            format_name="FP8_SOURCE",
            source_kind="fp8",
            zero_cost_by_construction=False,
            serving_route=f"{ROUTE_DELEGATED_NATIVE}_fp8_block",
            route_status=ROUTE_STATUS_BACKED,
            detail=(
                "native block-FP8 with an FP32 weight_scale_inv plane; the "
                "cost pipeline emits real rows for it, so its candidate is "
                "not synthesized"
            ),
            route_evidence=(
                "stock compressed-tensors block-fp8; served on the "
                "checkpoints this format was written for (MiniMax M2, "
                "DeepSeek-V3)"
            ),
        ),
        # DeepSeek-V3.1/V4 block-FP8: one-byte UE8M0 block exponents. Same
        # element grid as FP8_SOURCE, different scale plane and different
        # byte count — see the FormatSpec comment.
        SourcePassthroughContract(
            format_name="FP8_BLOCK_UE8M0_SOURCE",
            source_kind="fp8_ue8m0",
            wire_format_id="fp8_e4m3_ue8m0_block128",
            zero_cost_by_construction=True,
            serving_route=f"{ROUTE_DELEGATED_NATIVE}_fp8_block_ue8m0",
            route_status=ROUTE_STATUS_BLOCKED,
            detail=(
                "native block-FP8 with UE8M0 block exponents. 'The checkpoint "
                "already serves this way' turned out NOT to imply a serve "
                "route on our target: measured on GB10/sm121, every route is "
                "dead. Keeping the rung on the menu is deliberate — if the DP "
                "still wants it, that is the serving gap becoming visible in "
                "the allocation instead of at deploy time."
            ),
            route_evidence=(
                "sm121 measured 2026-08-03: deep_gemm assert; cutlass "
                "scaled_mm rejects the block layout; triton KeyError on "
                "float8_e8m0fnu; flashinfer gated sm90-exact; marlin-linear "
                "is <=sm89. CONSEQUENCE: CB re-encoding of the body is the "
                "only way this checkpoint serves on this box at all."
            ),
        ),
        # DeepSeek-V4 routed experts: nibble-packed E2M1 + E8M0 group scales.
        SourcePassthroughContract(
            format_name="MXFP4_SOURCE",
            source_kind="mxfp4",
            wire_format_id="mxfp4_e2m1_ue8m0_g32",
            zero_cost_by_construction=True,
            serving_route=f"{ROUTE_DELEGATED_NATIVE}_mxfp4",
            route_status=ROUTE_STATUS_BACKED,
            route_requirement="vllm --moe-backend marlin",
            detail=(
                "native packed MXFP4 routed experts, served by the model's "
                "own path rather than a codebook decoder. BACKED, but only "
                "off the default: the auto-selected backend is broken on our "
                "target, so the requirement below is part of the contract, "
                "not a tuning hint."
            ),
            route_evidence=(
                "sm121 measured 2026-08-03: native-confirmed via vLLM Marlin "
                "MoE (--moe-backend marlin). The AUTO default DeepGEMM_MXFP4 "
                "asserts on the SF transformation; FlashInfer is gated to "
                "capability family 100; the OAI Triton path is hard-excluded "
                "on SM12x (0/15 kernels)."
            ),
        ),
        # Unquantized passthrough. Not a "source format" in the census sense,
        # but it obeys the same rule and predates this table.
        SourcePassthroughContract(
            format_name="BF16",
            source_kind="bf16",
            zero_cost_by_construction=False,
            serving_route=f"{ROUTE_DELEGATED_NATIVE}_bf16",
            route_status=ROUTE_STATUS_BACKED,
            detail="unquantized passthrough on a bf16 source",
            route_evidence="plain container floats; no kernel required",
        ),
    )
}

# Kept as the flat {format: required_kind} view every existing consumer
# (allocator promotion legality, export defensive checks, the visual
# passthrough contract) already imports. DERIVED from the table above so a new
# census format cannot be legal in one place and unknown in the other.
PASSTHROUGH_SOURCE_REQUIREMENTS: dict[str, str] = {
    name: contract.source_kind
    for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items()
}

# Passthrough formats whose Δloss is EXACT BY CONSTRUCTION, so the allocator
# synthesizes their candidate instead of requiring a cost-table column.
#
# Every cost in this pipeline is measured against the DEQUANTIZED SOURCE. A
# format that ships the source bytes UNCHANGED is therefore the identity
# transform on the reference: its Δloss is 0.0 as a matter of arithmetic, not
# of measurement, and no cost run can ever produce a more accurate number for
# it. FP8_SOURCE and BF16 are deliberately NOT members: the cost pipeline
# already emits real rows for them on the checkpoints where they are legal,
# and synthesizing over a table that has an entry would hide a disagreement.
# The newer source formats are members because no cost table will ever have a
# column for them — they are byte-copy contracts, and asking the encoder to
# "measure" a copy would burn GPU hours to reproduce a zero.
#
# Membership is not self-certifying: ``cost_entry_is_source_passthrough``
# additionally requires the format to be a registered passthrough format whose
# activation path is the identity, so a stray ``cost_source`` string in a
# hand-written table cannot claim exactness for an activation-quantizing rung.
SOURCE_PASSTHROUGH_FORMATS: frozenset[str] = frozenset(
    name for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items()
    if contract.zero_cost_by_construction
)

# Passthrough formats with no DEMONSTRATED serve route on the target — either
# unaudited (pending) or measured dead (blocked). On the menu (an allocator
# that wants one is reporting a serving gap), but the exporter refuses to ship
# a selection containing one without an explicit override.
ROUTE_PENDING_PASSTHROUGH_FORMATS: frozenset[str] = frozenset(
    name for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items()
    if not contract.route_backed
)

# The closed wire enum the artifact declares and the serving side reads
# (quant_config.json "source_passthrough"). Registry names are ours to rename;
# these are a cross-repo contract, so they are declared once here and the
# exporter maps through this table rather than spelling them again.
PASSTHROUGH_WIRE_FORMAT_IDS: dict[str, str] = {
    name: contract.wire_format_id
    for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items()
    if contract.wire_format_id is not None
}

# The same cross-repo wire enum, for formats this producer RE-ENCODES rather
# than copies. Kept as its own table beside the passthrough ids rather than
# folded into SOURCE_PASSTHROUGH_CONTRACTS, because that table means something
# specific and load-bearing: every member has an IDENTITY codec and a Δloss of
# zero by construction. A re-quantization rung has a real encoder and a real
# measured cost, so declaring it there would let it claim an exactness it has
# not earned (tests/test_source_passthrough_family.py pins that invariant).
#
# What the two tables DO share is that the id is a contract with the consumer:
# the registry name is ours to rename, the wire id is not.
#
# NOTE FOR ORCHESTRATOR RECONCILIATION: ``mxfp8_e4m3_e8m0_g32`` is the
# spelling the Gridbook consumer side proposed. It diverges from this repo's
# own convention, which spells the unsigned-E8M0 scale plane ``ue8m0``
# (``mxfp4_e2m1_ue8m0_g32``, ``fp8_e4m3_ue8m0_block128``). The consumer's
# spelling is used verbatim here rather than guessed at; if the two repos
# settle on ``mxfp8_e4m3_ue8m0_g32`` instead, this one string is the only
# producer-side edit.
REQUANT_WIRE_FORMAT_IDS: dict[str, str] = {
    "MXFP8_UE8M0_G32": "mxfp8_e4m3_e8m0_g32",
}

# Every wire id this producer can declare, from either table. Ids must be
# globally unique: the consumer dispatches on the id alone and cannot tell
# which producer-side table it came from.
WIRE_FORMAT_IDS: dict[str, str] = {
    **PASSTHROUGH_WIRE_FORMAT_IDS,
    **REQUANT_WIRE_FORMAT_IDS,
}


def passthrough_serving_notes() -> dict[str, dict[str, str]]:
    """Per-format serving requirements/evidence for the artifact's notes.

    A BACKED route with an unmet requirement serves no better than a blocked
    one, so the requirement travels with the artifact rather than living in a
    run book.
    """
    return {
        name: {
            key: value
            for key, value in (
                ("route_status", contract.route_status),
                ("route", contract.serving_route),
                ("requirement", contract.route_requirement),
                ("evidence", contract.route_evidence),
            )
            if value is not None
        }
        for name, contract in sorted(SOURCE_PASSTHROUGH_CONTRACTS.items())
    }


def serialized_candidate_payload(
    spec: fr.FormatSpec,
    shape: tuple[int, ...],
    *,
    qname: str,
    cb_serialization_context: CBSerializationContext | None,
) -> tuple[int, str | None, str | None]:
    """Return producer payload bytes + identity for one candidate tensor.

    A CB FormatSpec intentionally describes only the historical nominal body;
    it cannot encode layout-v1/v2, FP8 row scales, or shared codebook identity.
    Those formats must use the versioned producer accountant.  Refusing a
    missing context prevents an old ``4k+16`` estimate from silently leaking
    back into a production-v2 allocation.
    """
    if not is_cb_format(spec.name):
        return int(spec.memory_bytes_for_shape(shape)), None, None
    if cb_serialization_context is None:
        raise ValueError(
            f"{qname}: exact bytes for {spec.name} require an explicit "
            "CBSerializationContext (scale coding/layout + codebook identity); "
            "refusing to price the legacy FormatSpec approximation"
        )
    item = cb_tensor_payload_breakdown(
        spec.name,
        shape,
        qname=qname,
        context=cb_serialization_context,
    )
    return (
        int(item["tensor_payload_bytes"]),
        (str(item["identity_key"])
         if item.get("identity_key") is not None else None),
        (str(item["sidecar_identity_key"])
         if item.get("sidecar_identity_key") is not None else None),
    )


def _is_passthrough_format(format_name: str) -> bool:
    return format_name in PASSTHROUGH_SOURCE_REQUIREMENTS


def _passthrough_source_ok(
    format_name: str,
    source_kind: str | None,
) -> bool:
    required = PASSTHROUGH_SOURCE_REQUIREMENTS.get(format_name)
    if required is None:
        return True
    if source_kind is None:
        return format_name == "BF16"
    return source_kind == required


@dataclass(frozen=True)
class FormatApplicability:
    legal: bool
    reason: str | None = None
    detail: str = ""


def _profile_allows_format(
    target_profile: str | None,
    name: str | None,
    fmt: str,
    packed_expert: bool | None = None,
) -> FormatApplicability:
    decision = check_serving_format(target_profile, name, fmt,
                                    packed_expert=packed_expert)
    return FormatApplicability(
        decision.legal,
        decision.reason,
        decision.detail,
    )


def _format_kernel_supports_shape(fmt_name: str, in_features: int,
                                  out_features: int) -> bool:
    """Return True if the runtime kernel can handle this Linear shape."""
    return check_serving_shape(
        "research",
        fmt_name,
        in_features=in_features,
        out_features=out_features,
    ).legal


def check_format_applicability(
    linear_shape: tuple[int, ...],
    format_spec_or_name: fr.FormatSpec | str,
    *,
    qname: str | None = None,
    source_kind: str | None = None,
    target_profile: str | None = None,
) -> FormatApplicability:
    """Return whether a Linear shape can legally use a format.

    The verdict captures all cheap preflight constraints that otherwise show
    up later as allocator-invalid choices or RTN/kernel crashes: source
    passthrough integrity, serving profile restrictions, group divisibility,
    and known runtime kernel shape rules.
    """
    try:
        spec = (
            format_spec_or_name
            if isinstance(format_spec_or_name, fr.FormatSpec)
            else fr.get_format(str(format_spec_or_name))
        )
    except KeyError as exc:
        return FormatApplicability(False, "unknown_format", str(exc))
    fmt = fr.canonical_format_name(spec.name)
    shape = tuple(int(dim) for dim in linear_shape)
    if len(shape) < 2:
        return FormatApplicability(
            False,
            "shape_rank",
            f"expected a Linear weight shape with rank >= 2, got {shape}",
        )
    out_features = int(shape[-2])
    in_features = int(shape[-1])

    if (
        _is_passthrough_format(fmt)
        and not _passthrough_source_ok(fmt, source_kind)
    ):
        required = PASSTHROUGH_SOURCE_REQUIREMENTS.get(fmt)
        return FormatApplicability(
            False,
            "source_dtype_mismatch",
            f"{fmt} requires source_kind={required!r}, got {source_kind!r}",
        )

    # Rank-3 shapes ARE packed expert stacks — the profile can scope rules
    # to them (containers whose stock-CT delegation is dense-only).
    profile_verdict = _profile_allows_format(
        target_profile, qname, fmt, packed_expert=len(shape) >= 3)
    if not profile_verdict.legal:
        return profile_verdict

    if (
        spec.group_size > 0
        and int(spec.group_size) < in_features
        and in_features % int(spec.group_size) != 0
    ):
        return FormatApplicability(
            False,
            "group_divisibility",
            f"group_size={spec.group_size} does not divide in_features="
            f"{in_features}",
        )
    if spec.scale_block_shape is not None:
        block_rows, block_cols = spec.scale_block_shape
        if out_features % int(block_rows) != 0 or in_features % int(block_cols) != 0:
            return FormatApplicability(
                False,
                "scale_block_divisibility",
                f"scale_block_shape={spec.scale_block_shape} does not divide "
                f"(out_features={out_features}, in_features={in_features})",
            )

    shape_decision = check_serving_shape(
        target_profile,
        fmt,
        qname=qname,
        in_features=in_features,
        out_features=out_features,
    )
    if not shape_decision.legal:
        return FormatApplicability(
            False,
            shape_decision.reason or "kernel_shape",
            shape_decision.detail,
        )
    return FormatApplicability(True)


def check_stats_format_applicability(
    stats_entry: dict,
    format_spec_or_name: fr.FormatSpec | str,
    *,
    qname: str | None = None,
    source_kind: str | None = None,
    target_profile: str | None = None,
) -> FormatApplicability:
    """Stats-entry wrapper for ``check_format_applicability``.

    This is the path allocator-like code should use when it only has the
    probe stats table.  Rank-1 legacy stats do not carry enough shape
    information for kernel preflight, so they remain admissible and the
    exporter keeps the final safety check.
    """
    shape = _shape_from_stats(dict(stats_entry))
    if len(shape) < 2:
        return FormatApplicability(True)
    return check_format_applicability(
        shape,
        format_spec_or_name,
        qname=qname,
        source_kind=source_kind,
        target_profile=target_profile,
    )


def _flashinfer_kernel_accepts(fmt_name: str, in_features: int,
                               out_features: int) -> bool | None:
    """Compatibility wrapper for the config-backed FlashInfer validator."""
    from .runtime_shape_validators import flashinfer_mxfp8_problem_size_accepts

    return flashinfer_mxfp8_problem_size_accepts(
        fmt_name,
        in_features=in_features,
        out_features=out_features,
    )


def _stats_indicates_packed_expert(stats_entry: dict) -> bool:
    """True for probe entries representing a 3D packed-expert tensor."""
    return bool(
        stats_entry.get("_packed_experts_module")
        or stats_entry.get("_packed_param")
        or int(stats_entry.get("num_experts", 0) or 0) > 0
    )


def _has_measured_output_mse(stats_entry: dict, cost_entry: dict) -> bool:
    """Whether ``output_mse`` is a real joint-output measurement.

    Packed experts historically stored ``output_mse=0.0`` as a placeholder
    because the routed expert forward was not reconstructed offline. That
    placeholder must not outrank the scalar predicted_dloss / weight_mse path.
    """
    if "output_mse" not in cost_entry:
        return False
    if cost_entry.get("output_mse_measured") is False:
        return False
    if (_stats_indicates_packed_expert(stats_entry)
            and float(cost_entry.get("output_mse", 0.0)) == 0.0
            and ("predicted_dloss" in cost_entry or "weight_mse" in cost_entry)):
        return False
    return True


def cost_entry_is_bit_exact(
    cost_entry: dict,
    format_name: str | None = None,
) -> bool:
    """Whether this entry proves a LOSSLESS re-encode end to end: measured
    ``weight_mse`` of exactly 0.0 AND a format whose activation path is the
    identity.

    ``weight_mse`` is a mean of squared per-element deltas: it is exactly
    zero only when the format stores the source weights verbatim (W' == W)
    — e.g. MXFP8 over an FP8 128-block source, or MXFP4/MXFP6/MXFP8 over
    an MXFP4-packed QAT source. But W' == W only silences the WEIGHT side.
    For W·A· formats (``FormatSpec.act_quant_changes_input`` — NVFP4,
    FP8 dynamic, the MX family, GGUF Q8_1 compute) the cost pipeline
    applies ``activation_quantize_dequantize(X)`` before measuring
    ``output_mse`` (measure_quant_cost), so a weight-lossless entry's
    output_mse is REAL A-side error, not noise — on an MXFP4-packed
    source, an MXFP4 re-encode priced from weight_mse alone would cost
    dloss 0.0, the unbeatable global minimum at any budget, while its
    served activations are still 4-bit. The short-circuit therefore
    requires the format's activation quantization to be the identity — a
    dtype-level fact (``FormatSpec.act_quant_changes_input``, i.e. ``act_bits``
    absent or >= 16: BF16, FP8_SOURCE, NVFP4A16, MXFP8A16, INT-W·A16), not a
    heuristic. Formats we cannot identify
    (``format_name`` None or unregistered) never short-circuit.

    For qualifying passthrough-activation formats, measured zero is a
    valid, indeed optimal, cost (see ``_log_error_values`` in
    allocator.py): the entry short-circuits to predicted dloss 0.0 ahead
    of any noisy output_mse measurement.

    Entries that declare an explicit ``cost_source`` (e.g. the
    production-render score pipeline) carry their own authoritative
    pricing and default ``weight_mse`` to 0.0 as a placeholder, not a
    measurement — they are never treated as bit-exact, matching the
    precedence ``cost_entry_source`` already gives the explicit source.
    """
    explicit = cost_entry.get("cost_source")
    if isinstance(explicit, str) and explicit:
        return False
    if format_name is None:
        return False
    try:
        spec = fr.get_format(str(format_name))
    except KeyError:
        return False
    if spec.act_quant_changes_input:
        return False
    weight_mse = cost_entry.get("weight_mse")
    try:
        return weight_mse is not None and float(weight_mse) == 0.0
    except (TypeError, ValueError):
        return False


def cost_entry_is_source_passthrough(
    cost_entry: dict,
    format_name: str | None = None,
) -> bool:
    """Whether this entry ships the SOURCE bytes and is therefore exact.

    The dual of ``cost_entry_is_bit_exact``. That predicate proves exactness
    from a MEASUREMENT (``weight_mse == 0.0`` recorded by a real cost run);
    this one proves it from a CONTRACT — the exporter copies the source slice
    verbatim, so the re-encode error is not small, it does not exist.

    Three independent conditions, all required, so the claim cannot be forged
    by writing a string into a cost table:

      * the entry declares ``cost_source="source_passthrough"``;
      * the format is a declared member of ``SOURCE_PASSTHROUGH_FORMATS``;
      * the format's activation path is the identity
        (``FormatSpec.act_quant_changes_input`` is False) — the same
        dtype-level gate ``cost_entry_is_bit_exact`` applies, because
        shipping the weights verbatim only silences the W side.

    ``cost_entry_is_bit_exact`` deliberately refuses ANY entry carrying an
    explicit ``cost_source``, since those normally mean "an upstream pipeline
    priced this row and defaulted weight_mse to a placeholder 0.0". A
    source-passthrough entry is the one case where the explicit provenance is
    itself the proof, which is why it gets a predicate of its own rather than
    a hole punched in that rule.
    """
    if cost_entry.get("cost_source") != SOURCE_PASSTHROUGH_COST_SOURCE:
        return False
    if format_name is None:
        return False
    canonical = fr.canonical_format_name(str(format_name))
    if canonical not in SOURCE_PASSTHROUGH_FORMATS:
        return False
    try:
        spec = fr.get_format(canonical)
    except KeyError:
        return False
    return not spec.act_quant_changes_input


BAND_INTERPOLATED_COST_SOURCE = "band_interpolated"


def cost_entry_is_band_interpolated(cost_entry: dict) -> bool:
    """Whether this row's cost was FITTED from ladder anchors, not measured.

    Stamped by the cost stage's RD-ladder interpolation
    (``measure_quant_cost``, ``PRISMAQUANT_CB_LADDER_INTERP=1``). Such a row
    is not a guess — the tensor's own law had to clear a holdout gate before
    the fit was accepted, and a tensor whose law was rejected had its rungs
    measured instead — but it IS a prediction, and a shipped artifact must be
    able to say which of its selected prices were predictions.
    """
    return cost_entry.get("cost_source") == BAND_INTERPOLATED_COST_SOURCE


def drop_interpolated_candidates_dominated_by_measured(
    candidates: dict[str, list[Candidate]],
    costs: dict,
    *,
    band: float,
) -> tuple[dict[str, list[Candidate]], int]:
    """Remove interpolated candidates a measured one already beats.

    The DP optimizes over estimates, so a Δloss gap SMALLER than the
    interpolator's own validated error is not evidence — it is a coin flip,
    and letting it decide means an unmeasured rung can displace a measured one
    on noise alone. This drops an interpolated candidate only when a measured
    candidate for the SAME unit is both

      * within ``band`` relative Δloss (i.e. indistinguishable at the
        interpolator's demonstrated resolution), and
      * no more expensive in bytes.

    Both conditions are required, so a genuine trade survives: an interpolated
    rung that is materially cheaper in bytes, or materially better in Δloss,
    is still on the menu. Only the strictly-dominated-within-noise case is
    removed. ``band`` must be the MEASURED holdout error from this run's own
    validation, not a taste constant.

    Returns the filtered candidates and the number dropped.
    """
    if band <= 0.0:
        return candidates, 0
    out: dict[str, list[Candidate]] = {}
    dropped = 0
    for name, cands in candidates.items():
        rows = costs.get(name, {})
        measured = [
            c for c in cands
            if not cost_entry_is_band_interpolated(rows.get(c.fmt, {}))
        ]
        kept = []
        for cand in cands:
            if not cost_entry_is_band_interpolated(rows.get(cand.fmt, {})):
                kept.append(cand)
                continue
            scale = max(abs(cand.predicted_dloss), 1e-30)
            if any(
                other.memory_bytes <= cand.memory_bytes
                and abs(other.predicted_dloss - cand.predicted_dloss) / scale
                <= band
                for other in measured
            ):
                dropped += 1
                continue
            kept.append(cand)
        out[name] = kept or cands
    return out, dropped


def cost_entry_is_exact_by_construction(
    cost_entry: dict,
    format_name: str | None = None,
) -> bool:
    """Whether this row's Δloss is exactly 0.0 with no measurement involved.

    The union of the two ways a row can be free: a measured lossless
    re-encode (``cost_entry_is_bit_exact``) and a byte-verbatim source
    passthrough (``cost_entry_is_source_passthrough``). Pricing, the measured
    branch test and the P5a branch label all key off THIS predicate so the two
    cannot drift apart; ``cost_entry_source`` still reports which of the two
    it was.
    """
    return (
        cost_entry_is_bit_exact(cost_entry, format_name)
        or cost_entry_is_source_passthrough(cost_entry, format_name)
    )


def synthesized_source_passthrough_cost_entry(format_name: str) -> dict:
    """The cost row for a passthrough candidate the cost table cannot hold.

    Not a placeholder and not an estimate: ``predicted_dloss`` is 0.0 because
    the exporter ships the source slice unchanged, and the explicit
    ``cost_source`` records that provenance so no downstream reader mistakes
    the zero for an unmeasured activation cost (the failure mode
    ``cost_entry_prices_unmeasured_activation_at_zero`` exists to catch).
    ``output_mse_measured=False`` states plainly that no output measurement
    was taken — because none was needed.
    """
    return {
        "cost_source": SOURCE_PASSTHROUGH_COST_SOURCE,
        "predicted_dloss": 0.0,
        "weight_mse": 0.0,
        "output_mse": 0.0,
        "output_mse_measured": False,
        "source_passthrough_format": fr.canonical_format_name(
            str(format_name)
        ),
    }


def cost_entry_uses_measured_output_mse(
    stats_entry: dict,
    cost_entry: dict,
    format_name: str | None = None,
) -> bool:
    """Whether ``cost_entry_predicted_dloss`` will read ``output_mse``."""
    if cost_entry_is_exact_by_construction(cost_entry, format_name):
        return False
    return _has_measured_output_mse(stats_entry, cost_entry)


def cost_entry_source(
    stats_entry: dict,
    cost_entry: dict,
    format_name: str | None = None,
) -> str:
    """Return the named cost source the allocator will use for one row."""
    explicit = cost_entry.get("cost_source")
    if isinstance(explicit, str) and explicit:
        return explicit
    if cost_entry_is_bit_exact(cost_entry, format_name):
        return "bit_exact"
    if _has_measured_output_mse(stats_entry, cost_entry):
        if (
            _fisher_output_mse_allocator_enabled()
            and "fisher_output_mse" in cost_entry
        ):
            return "fisher_output_mse"
        return "output_mse"
    if "predicted_dloss" in cost_entry:
        return "predicted_dloss"
    return "weight_mse"


def _fisher_output_mse_allocator_enabled() -> bool:
    value = os.environ.get("PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR")
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def cost_entry_measured_activation_dloss(
    stats_entry: dict,
    cost_entry: dict,
    *,
    gain: float = 1.0,
) -> float:
    """The ACTIVATION-INCLUSIVE price of one row, or 0.0 when unmeasured.

    Split out of ``cost_entry_predicted_dloss`` so the P5a calibration reads
    exactly the number the measured branch would have priced — one
    implementation of "what does the output_mse branch say", not a second
    copy that can drift from the precedence chain it calibrates against.
    ``measure_quant_cost`` applies ``activation_quantize_dequantize(X)``
    before measuring ``output_mse``, which is why this branch — and only this
    branch — carries the A side.
    """
    if (
        _fisher_output_mse_allocator_enabled()
        and "fisher_output_mse" in cost_entry
    ):
        return predicted_dloss(
            stats_entry["h_trace"],
            float(cost_entry["fisher_output_mse"]),
            gain=gain,
        )
    return predicted_dloss(
        stats_entry["h_trace"],
        float(cost_entry.get("output_mse", 0.0)),
        gain=gain,
    )


def cost_entry_weight_only_dloss(
    stats_entry: dict,
    cost_entry: dict,
    *,
    gain: float = 1.0,
) -> float:
    """The WEIGHT-ONLY price of one row (``predicted_dloss``/``weight_mse``).

    Uncorrected by any activation calibration — this is the number the
    calibration divides into, and the number the correction multiplies.
    """
    if "predicted_dloss" in cost_entry:
        base = float(cost_entry["predicted_dloss"])
        # Uncertainty-aware allocation (opt-in): charge z·stderr on top of the
        # point estimate. The knapsack optimizes over noisy estimates, so it
        # systematically harvests lucky draws (winner's curse — the observed
        # ±0.017-KL between-seed allocation lottery). UCB takes an aggressive
        # format choice only when it is CONFIDENTLY cheap: a non-regressive
        # bias whose penalty is derived from the measurement's own sampling
        # noise, not a tuned constant. z=0 (default) is bit-identical to
        # prior behavior.
        z = _cost_ucb_z()
        if z > 0.0:
            base += z * float(cost_entry.get("predicted_dloss_stderr", 0.0))
        return base * float(gain)
    return predicted_dloss(
        stats_entry["h_trace"],
        float(cost_entry.get("weight_mse", 0.0)),
        gain=gain,
    )


def cost_entry_activation_pricing_branch(
    stats_entry: dict,
    cost_entry: dict,
    format_name: str | None = None,
    activation_pricing: ActivationFairPricing | None = None,
) -> str:
    """Name the estimator that priced one row's ACTIVATION contract.

    Orthogonal to ``cost_entry_source`` (which names the cost *field*): this
    answers the audit's question — did this row's price ever see the A side,
    and if not, was it corrected? Stamped on every ``Candidate`` so the
    question is answerable from the artifact rather than from the code
    version that produced it.
    """
    if cost_entry_is_source_passthrough(cost_entry, format_name):
        # Neither measured nor weight-only: the row was never priced from an
        # error estimate at all. It gets its own label so the artifact can
        # tell "shipped the source bytes" apart from "measured a lossless
        # re-encode" — both are free, but only one of them ran an encoder.
        return BRANCH_SOURCE_PASSTHROUGH
    if cost_entry_is_bit_exact(cost_entry, format_name):
        return BRANCH_BIT_EXACT
    if _has_measured_output_mse(stats_entry, cost_entry):
        return BRANCH_MEASURED
    if cost_entry.get(APPLIED_MARKER_KEY) is True:
        # An aggregated super-item entry: the members' penalties are already
        # folded into its predicted_dloss (aggregate_* below).
        return BRANCH_CALIBRATED
    act_changes = _format_act_quant_changes_input(format_name)
    if activation_pricing is None:
        return (
            BRANCH_ACTIVATION_IDENTITY if not act_changes
            else BRANCH_UNCALIBRATED
        )
    return activation_pricing.penalty_for(format_name, act_changes)[1]


def _format_act_quant_changes_input(format_name: str | None) -> bool:
    if format_name is None:
        return False
    try:
        return bool(fr.get_format(str(format_name)).act_quant_changes_input)
    except KeyError:
        return False


def _activation_penalty(
    format_name: str | None,
    activation_pricing: ActivationFairPricing | None,
) -> float:
    if activation_pricing is None:
        return 1.0
    return activation_pricing.penalty_for(
        format_name, _format_act_quant_changes_input(format_name))[0]


def cost_entry_predicted_dloss(
    stats_entry: dict,
    cost_entry: dict,
    *,
    gain: float = 1.0,
    format_name: str | None = None,
    activation_pricing: ActivationFairPricing | None = None,
) -> float:
    """Return the allocator's authoritative Δloss for one cost entry.

    ``activation_pricing`` (P5a) applies the per-family activation calibration
    to the WEIGHT-ONLY branches only — the measured branch is already
    activation-inclusive and the bit-exact branch is, by construction, an
    identity activation path. ``None`` (the default) is bit-for-bit the
    pre-P5a precedence, which is what ``kl_measurement`` and every direct
    caller outside candidate construction still want.

    The correction is multiplicative, so it scales the UCB hedge with the
    point estimate (both are in the same weight-only units and the transfer
    to the measured scale applies to both), and it cannot lift an exactly-0.0
    price off zero — ``cost_entry_prices_unmeasured_activation_at_zero``
    keeps its full strength.
    """
    if cost_entry_is_exact_by_construction(cost_entry, format_name):
        # Zero cost by construction, in one of two ways: a lossless re-encode
        # END TO END (weights verbatim AND identity activation path,
        # ``cost_entry_is_bit_exact``), or a byte-verbatim source passthrough
        # whose exporter never re-encodes at all
        # (``cost_entry_is_source_passthrough``). Either way the answer does
        # not depend on a measurement, so it outranks any noisy output_mse.
        return 0.0
    if _has_measured_output_mse(stats_entry, cost_entry):
        return cost_entry_measured_activation_dloss(
            stats_entry, cost_entry, gain=gain)
    base = cost_entry_weight_only_dloss(stats_entry, cost_entry, gain=gain)
    if cost_entry.get(APPLIED_MARKER_KEY) is True:
        # Aggregated super item: its members were penalized individually and
        # the result summed. Re-applying here would square the correction.
        return base
    return base * _activation_penalty(format_name, activation_pricing)


ACTIVATION_COST_UNMEASURED_REASON = "activation_cost_unmeasured"


def cost_entry_prices_unmeasured_activation_at_zero(
    stats_entry: dict,
    cost_entry: dict,
    priced_dloss: float,
    format_name: str | None = None,
) -> bool:
    """Whether this row prices a W-and-A format's UNKNOWN cost at the global
    optimum (Δloss exactly 0.0).

    ``cost_entry_is_bit_exact`` closed this for the ``output_mse`` branch: a
    weight-lossless entry on an activation-quantizing format keeps its measured
    A-side output_mse instead of short-circuiting to 0.0. But that branch is
    only taken when output_mse is a real measurement. Packed-expert rows whose
    routed forward could not be reconstructed (``can_measure_output`` false),
    and EVERY row in a run with ``PRISMAQUANT_EXPERT_COST_SAMPLE`` set, are
    written with ``output_mse_measured=False`` (measure_quant_cost), so pricing
    falls through to ``predicted_dloss``/``weight_mse`` — both of which are
    exactly 0.0 for a weight-lossless re-encode (the source is already in that
    format: MXFP4 over an MXFP4-packed source, NVFP4 over an NVFP4-CB source).
    Nothing in that row ever looked at the activation path, yet the DP reads a
    cost of 0.0: the unbeatable global minimum at any budget, for an assignment
    whose served activations are 4-bit.

    This is not a mis-estimate to be corrected — a positive weight-side
    surrogate is the accepted L1 design, biased but tradeable — it is a cost the
    optimizer CANNOT trade off: zero is the argmin, so the format is selected at
    every target, unconditionally. The unknown must therefore be excluded from
    the menu (``build_candidates``, counted and logged like any other
    inapplicable format) rather than priced.

    The predicate is exact, not thresholded:

      * the format's activation path is provably non-identity
        (``FormatSpec.act_quant_changes_input`` — a dtype-level fact);
      * no measured output-side evidence exists for this row
        (``_has_measured_output_mse``), so the A-side error is unknown
        whatever produced the number;
      * the resulting price is exactly 0.0 — the DP's global optimum;
      * the row's measured sensitivity is POSITIVE. ``h_trace == 0`` prices
        every format at 0.0 including the passthrough ones, which is a measured
        statement that no perturbation of this Linear's output moves the loss —
        W-side or A-side, since the same Fisher expansion multiplies both. A
        zero-token expert at thin calibration is exactly that row, and it must
        stay free to take the cheapest format instead of being forced onto
        BF16.
    """
    if format_name is None:
        return False
    try:
        spec = fr.get_format(str(format_name))
    except KeyError:
        return False
    if not spec.act_quant_changes_input:
        return False
    if _has_measured_output_mse(stats_entry, cost_entry):
        return False
    try:
        if float(priced_dloss) != 0.0:
            return False
    except (TypeError, ValueError):
        return False
    try:
        h_trace = float(stats_entry.get("h_trace", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return h_trace > 0.0


def collect_activation_calibration_rows(
    stats: dict,
    costs: dict,
    formats: list[fr.FormatSpec],
) -> tuple[list[CalibrationRow], dict[str, int], dict[str, int]]:
    """Extract the P5a calibration sample from the run's own cost tables.

    Returns ``(rows, measured_rows_by_family, weight_only_rows_by_family)``
    over ACTIVATION-QUANTIZING formats only (``act_quant_changes_input``): a
    passthrough/A16 rung has no A side to transfer, and its measured-vs-
    weight-only disagreement is a different question.

    A row joins the calibration sample when it carries BOTH estimators —
    a real measured ``output_mse`` (``_has_measured_output_mse``) AND a
    weight-only field, both strictly positive. Exactly-zero prices are
    excluded on both sides: a zero denominator has no ratio, and a zero
    measured price is the lossless-re-encode case the bit-exact
    short-circuit and ``cost_entry_prices_unmeasured_activation_at_zero``
    already own.

    Everything is computed at ``gain=1.0`` (the ratio is gain-invariant) and
    the iteration order is the sorted cost table, so the sample — and its
    digest — is deterministic.
    """
    rows: list[CalibrationRow] = []
    measured_by_family: dict[str, int] = {}
    weight_only_by_family: dict[str, int] = {}
    for spec in formats:
        if not spec.act_quant_changes_input:
            continue
        family = str(spec.family)
        measured_by_family.setdefault(family, 0)
        weight_only_by_family.setdefault(family, 0)
        for name in sorted(costs):
            stats_entry = stats.get(name)
            if not isinstance(stats_entry, dict):
                continue
            entry, _entry_fmt = _resolve_cost_entry(costs[name], spec.name)
            if entry is None or "error" in entry:
                continue
            if cost_entry_is_bit_exact(entry, spec.name):
                continue
            if _has_measured_output_mse(stats_entry, entry):
                measured_by_family[family] += 1
                if not ("predicted_dloss" in entry or "weight_mse" in entry):
                    continue
                measured = cost_entry_measured_activation_dloss(
                    stats_entry, entry)
                weight_only = cost_entry_weight_only_dloss(stats_entry, entry)
                if measured > 0.0 and weight_only > 0.0:
                    rows.append(CalibrationRow(
                        qname=str(name),
                        fmt=spec.name,
                        family=family,
                        measured_dloss=float(measured),
                        weight_only_dloss=float(weight_only),
                    ))
            else:
                weight_only_by_family[family] += 1
    return rows, measured_by_family, weight_only_by_family


def calibrate_activation_fair_pricing(
    stats: dict,
    costs: dict,
    formats: list[fr.FormatSpec],
    *,
    enabled: bool | None = None,
) -> ActivationFairPricing:
    """Calibrate the per-family activation penalty ONCE for a run.

    The allocator calls this before ``build_candidates`` and threads the
    result through every candidate-construction path, so body, MTP and visual
    menus share one fit (the audit's "calibrated once per model") instead of
    three menu-dependent ones.
    """
    rows, measured, weight_only = collect_activation_calibration_rows(
        stats, costs, formats)
    return _calibrate_activation_pricing(
        rows,
        measured_rows_by_family=measured,
        weight_only_rows_by_family=weight_only,
        enabled=enabled,
    )


def _cost_ucb_z() -> float:
    """PRISMAQUANT_COST_UCB_Z: stderr multiples added to predicted_dloss."""
    try:
        return max(0.0, float(os.environ.get("PRISMAQUANT_COST_UCB_Z", "0")))
    except Exception:
        return 0.0


def _resolve_cost_entry(cost_rows: dict, fmt_name: str) -> tuple[dict | None, str]:
    """Resolve one Linear's cost row for ``fmt_name``, alias-aware.

    Returns ``(entry, entry_fmt)`` where ``entry_fmt`` is the alias actually
    present in the cost table (what ``calibrated_gains`` may be keyed by), or
    ``(None, fmt_name)`` when the format was never measured.
    """
    for candidate_name in fr.aliases_for(fmt_name):
        if candidate_name in cost_rows:
            return cost_rows[candidate_name], candidate_name
    return None, fmt_name


def _super_item_ucb_hedge(member_terms, ucb_z: float) -> tuple[float, float]:
    """Convert a super item's per-member UCB hedge into the independence one.

    A super item (fused-sibling group, packed serving group) prices one
    format as the SUM of its members' ``cost_entry_predicted_dloss``. With
    ``PRISMAQUANT_COST_UCB_Z > 0`` every member term already carries its own
    ``z·stderr·gain``, so the sum carries a LINEAR ``z·Σ(stderr·gain)``
    hedge — up to a √N OVER-hedge on an N-member group. The member dloss
    estimates are independent measurements, so the stderr of the group SUM is
    ``sqrt(Σ (stderr·gain)²)``.

    ``member_terms`` yields ``(stats_entry, cost_entry, member_dloss, scale)``
    where ``scale`` is every multiplicative factor already applied to that
    member's dloss but NOT to the raw ``predicted_dloss_stderr`` in the cost
    row — the calibrated gain and, since ultraplan P5a, the per-family
    activation penalty. Both scale the point estimate and its stderr
    identically, so a member whose price was penalized must have its stderr
    penalized too or the conversion would over-subtract the linear hedge it
    exists to undo.

    Returns ``(hedge_linear, stderr_agg)``: subtract ``hedge_linear`` from the
    member sum and add ``ucb_z * stderr_agg`` to get the independence
    aggregate. At ``ucb_z == 0`` ``hedge_linear`` is exactly 0.0, so the
    conversion is a bit-for-bit identity on the sum.

    Both aggregation paths call this so the two constructions cannot drift.
    """
    stderr_eff_sq = 0.0
    hedge_linear = 0.0
    for stats_entry, cost_entry, member_dloss, gain in member_terms:
        if float(member_dloss) <= 0.0:
            # Bit-exact re-encode / clamped-at-zero member: contributed no
            # dloss (and no hedge) to the sum; skip it symmetrically.
            continue
        if cost_entry is None or "error" in cost_entry:
            continue
        # Mirror cost_entry_predicted_dloss: the stderr hedge is only applied
        # on the explicit predicted_dloss branch.
        if _has_measured_output_mse(stats_entry, cost_entry):
            continue
        if "predicted_dloss" not in cost_entry:
            continue
        try:
            stderr = float(cost_entry.get("predicted_dloss_stderr", 0.0) or 0.0)
        except (TypeError, ValueError):
            stderr = 0.0
        if stderr <= 0.0:
            continue
        stderr_eff_sq += (stderr * float(gain)) ** 2
        hedge_linear += ucb_z * stderr * float(gain)
    return hedge_linear, math.sqrt(stderr_eff_sq)


def build_candidates(stats: dict, costs: dict, formats: list[fr.FormatSpec],
                     calibrated_gains: dict[str, float] | None = None,
                     source_manifest: dict[str, str] | None = None,
                     target_profile: str | None = None,
                     mask_records: list[dict] | None = None,
                     cb_serialization_context: CBSerializationContext | None = None,
                     activation_pricing: ActivationFairPricing | None = None,
                     ) -> dict[str, list[Candidate]]:
    """Build runtime-legal format candidates for every measured Linear.

    This is the optimizer's first legality gate. Export keeps a final
    defensive check for stale or hand-written recipes, but the DP must never
    see choices that the selected serving profile cannot run.

    ``activation_pricing`` (ultraplan P5a) is the run's ONE per-family
    activation calibration; it corrects the weight-only branches and stamps
    the branch that priced every candidate. ``target_profile``'s declared
    serving lanes (P5b) are resolved once per format and attached to every
    candidate, so the concrete route — activation contract, whether the
    consumer's fused mid-M kernel backs this rung, fallback — travels WITH
    the choice instead of being reconstructed from the format name later.
    """
    gains = calibrated_gains or {}
    out: dict[str, list[Candidate]] = {}
    masked: dict[tuple[str, str], list[str]] = {}
    source_counts: Counter[str] = Counter()
    activation_branch_counts: Counter[str] = Counter()
    unpriceable: dict[str, list[str]] = {}
    lane_by_format: dict[str, object] = {
        spec.name: serving_lane_route(target_profile, spec.name)
        for spec in formats
    }
    for name, s in stats.items():
        if name not in costs:
            continue
        shape = _shape_from_stats(s)
        in_features = int(s.get("in_features", 0) or 0)
        out_features = int(s.get("out_features", 0) or 0)
        source_kind = (
            source_manifest.get(name, "unknown")
            if source_manifest is not None else None
        )
        cands = []
        for spec in formats:
            entry = None
            entry_fmt = spec.name
            for candidate_name in fr.aliases_for(spec.name):
                if candidate_name in costs[name]:
                    entry = costs[name][candidate_name]
                    entry_fmt = candidate_name
                    break
            if entry is None and spec.name in SOURCE_PASSTHROUGH_FORMATS:
                # No cost table will ever carry a column for a byte-copy
                # contract. Synthesize the row rather than dropping the
                # candidate: silently omitting it would take the unit's
                # CHEAPEST ZERO-ERROR option off the menu and leave the DP
                # choosing only among lossy re-encodes. The legality gate
                # below still has to agree the source IS this format.
                entry = synthesized_source_passthrough_cost_entry(spec.name)
                entry_fmt = spec.name
            if entry is None or "error" in entry:
                continue
            verdict = check_stats_format_applicability(
                s,
                spec,
                qname=name,
                source_kind=source_kind,
                target_profile=target_profile,
            )
            if not verdict.legal:
                if mask_records is not None:
                    mask_records.append({
                        "qname": name,
                        "format": spec.name,
                        "reason": verdict.reason or "not_applicable",
                        "detail": verdict.detail,
                        "shape": [out_features, in_features],
                        "out_features": out_features,
                        "in_features": in_features,
                        "source_kind": source_kind,
                    })
                masked.setdefault(
                    (spec.name, verdict.reason or "not_applicable"),
                    [],
                ).append(name)
                continue
            gain = float(gains.get(spec.name, gains.get(entry_fmt, 1.0)))
            # Always use measured joint output perturbation when available.
            # Packed experts can carry an unmeasured output_mse placeholder;
            # cost_entry_predicted_dloss falls back to predicted_dloss or
            # weight_mse for those entries.
            predicted = cost_entry_predicted_dloss(
                s, entry, gain=gain, format_name=spec.name,
                activation_pricing=activation_pricing)
            priced = max(predicted, 0.0)
            if cost_entry_prices_unmeasured_activation_at_zero(
                    s, entry, priced, spec.name):
                # The A-side cost of this W-and-A format was never measured for
                # this row, and the W side is lossless, so the only price the
                # cost table can offer is 0.0 — the DP's global optimum. An
                # unknown priced at the optimum is always selected: exclude the
                # candidate (counted + logged, exactly like an inapplicable
                # format) instead of letting the optimizer read a cost that no
                # measurement supports. See
                # cost_entry_prices_unmeasured_activation_at_zero.
                h_trace = float(s.get("h_trace", 0.0) or 0.0)
                detail = (
                    f"{spec.name} quantizes activations (act_bits="
                    f"{spec.act_bits}) but this row has no measured "
                    "output_mse, and its weight-side error is exactly 0.0 "
                    "(lossless re-encode of an already-"
                    f"{spec.name}-shaped source), so the only available "
                    f"price is dloss 0.0 with h_trace={h_trace:.6g} > 0: an "
                    "unmeasured activation cost at the DP's global minimum "
                    f"(cost_source="
                    f"{cost_entry_source(s, entry, spec.name)})"
                )
                if mask_records is not None:
                    mask_records.append({
                        "qname": name,
                        "format": spec.name,
                        "reason": ACTIVATION_COST_UNMEASURED_REASON,
                        "detail": detail,
                        "shape": [out_features, in_features],
                        "out_features": out_features,
                        "in_features": in_features,
                        "source_kind": source_kind,
                    })
                masked.setdefault(
                    (spec.name, ACTIVATION_COST_UNMEASURED_REASON),
                    [],
                ).append(name)
                unpriceable.setdefault(name, []).append(spec.name)
                continue
            source_counts[cost_entry_source(s, entry, spec.name)] += 1
            activation_branch = cost_entry_activation_pricing_branch(
                s, entry, spec.name, activation_pricing)
            activation_branch_counts[activation_branch] += 1
            (
                memory_bytes,
                serialized_identity,
                serialized_sidecar_identity,
            ) = serialized_candidate_payload(
                spec,
                shape,
                qname=name,
                cb_serialization_context=cb_serialization_context,
            )
            s.setdefault("_memory_bytes_by_format", {})[spec.name] = memory_bytes
            if serialized_identity is not None:
                s.setdefault("_serialized_identity_by_format", {})[
                    spec.name
                ] = serialized_identity
                s.setdefault("_serialized_sidecar_identity_by_format", {})[
                    spec.name
                ] = serialized_sidecar_identity
            lane = lane_by_format.get(spec.name)
            if lane is not None:
                s.setdefault("_serving_lane_by_format", {})[spec.name] = lane
            cands.append(Candidate(
                fmt=spec.name,
                bits_per_param=8.0 * memory_bytes / max(int(math.prod(shape)), 1),
                memory_bytes=memory_bytes,
                predicted_dloss=priced,
                serialized_identity=serialized_identity,
                serialized_sidecar_identity=serialized_sidecar_identity,
                activation_pricing=activation_branch,
                serving_lane=lane,
            ))
        if cands:
            out[name] = cands
    if masked:
        for (fmt, reason), names in sorted(masked.items()):
            print(
                f"[alloc] format-applicability: {len(names)} Linear(s) "
                f"dropped {fmt} reason={reason} (sample: {names[:3]})",
                flush=True,
            )
    if source_counts:
        summary = ", ".join(
            f"{source}={count}" for source, count in sorted(source_counts.items())
        )
        print(f"[alloc] cost-source usage: {summary}", flush=True)
    if activation_branch_counts:
        # The audit's core complaint made visible per run: how many priced
        # rows never saw an activation measurement, and of those how many the
        # per-family calibration corrected.
        summary = ", ".join(
            f"{branch}={count}"
            for branch, count in sorted(activation_branch_counts.items())
        )
        print(f"[alloc] activation-pricing branch: {summary}", flush=True)
    starved = sorted(n for n in unpriceable if n not in out)
    if starved:
        # Excluding the unmeasured-activation candidates left these Linears
        # with NO candidate at all. Dropping them is worse than the bug we just
        # fixed: a name absent from `out` never reaches the DP, so its bits and
        # bytes vanish from the bpp/footprint accounting and from serving-unit
        # membership — silently, with the export still emitting the tensor.
        # The allocator cannot price these rows, so it must not pretend to.
        detail = "\n".join(
            f"    {n}: unpriceable={sorted(unpriceable[n])} (other cost rows: "
            f"{sorted(set(costs.get(n, {})) - set(unpriceable[n]))})"
            for n in starved[:8]
        )
        raise AssertionError(
            f"{len(starved)} Linear(s) have no priceable format left after "
            "excluding activation-quantizing formats whose activation-side "
            "cost was never measured and whose weight-side error is exactly "
            f"0.0:\n{detail}\n"
            "Every legal format for these rows would have been priced at "
            "dloss 0.0 (the DP's global minimum) on no activation-path "
            "evidence, and omitting the rows would silently shrink the "
            "allocator's bit/disk accounting and serving-unit membership. "
            "Close the measurement gap instead: unset "
            "PRISMAQUANT_EXPERT_COST_SAMPLE (and make the expert activation "
            "cache available) so measure_quant_cost records output_mse with "
            "activation_quantize_dequantize applied, or include a rung whose "
            "activation path is the identity (BF16, FP8_SOURCE, NVFP4A16, "
            "MXFP8A16) in the format menu."
        )
    return out


def selection_serving_lane_provenance(
    assignment: dict[str, str],
    candidates: dict[str, list[Candidate]] | None = None,
    target_profile: str | None = None,
) -> dict:
    """Per-selected-unit serving-route + activation-pricing provenance (P5b).

    "Neither repo can price an unbacked lane" only holds if the shipped
    artifact says which lane every selected unit actually rides. This walks
    the FINAL (expanded) assignment and reports, per format and in aggregate:
    the activation contract, whether the consumer's fused mid-M kernel backs
    that rung, the fallback route it takes when it does not, and which
    estimator priced the unit's activation cost.

    Routes are read from the chosen ``Candidate`` where one exists — the
    candidate is the object the DP actually saw — and re-resolved from the
    target profile for expanded members of aggregated super items, which have
    no candidate of their own. The two agree by construction (the lane is a
    function of format and profile); the fallback exists so an expanded
    packed-expert assignment is not silently reported as laneless.
    """
    lane_cache: dict[str, object] = {}
    by_format: dict[str, dict] = {}
    branch_counts: Counter[str] = Counter()
    contract_counts: Counter[str] = Counter()
    backed_rungs: set[int] = set()
    fallback_rungs: set[int] = set()
    n_backed = n_fallback = n_no_lane = 0

    for name in sorted(assignment):
        fmt = str(assignment[name])
        lane = None
        branch = None
        for cand in (candidates or {}).get(name, ()):
            if cand.fmt == fmt:
                lane = cand.serving_lane
                branch = cand.activation_pricing
                break
        if lane is None:
            if fmt not in lane_cache:
                lane_cache[fmt] = serving_lane_route(target_profile, fmt)
            lane = lane_cache[fmt]
        branch_counts[str(branch) if branch else "unrecorded"] += 1
        if lane is None:
            n_no_lane += 1
            by_format.setdefault(fmt, {
                "format": fmt, "units": 0, "route": None})["units"] += 1
            continue
        row = by_format.setdefault(fmt, {
            "format": fmt, "units": 0, "route": lane.as_dict()})
        row["units"] += 1
        contract_counts[lane.activation_contract or "unspecified"] += 1
        if lane.fused_mid_m_backed:
            n_backed += 1
            if lane.rung is not None:
                backed_rungs.add(int(lane.rung))
        else:
            n_fallback += 1
            if lane.rung is not None:
                fallback_rungs.add(int(lane.rung))

    return {
        "schema": SERVING_LANE_SCHEMA,
        "target_profile": str(target_profile or "research"),
        "gridbook_runtime_version": gridbook_runtime_version(),
        "units_total": len(assignment),
        "units_on_backed_fused_mid_m_lane": n_backed,
        "units_on_fallback_route": n_fallback,
        "units_without_declared_lane": n_no_lane,
        "selected_rungs_fused_mid_m_backed": sorted(backed_rungs),
        "selected_rungs_on_fallback_route": sorted(fallback_rungs),
        "activation_contracts": dict(sorted(contract_counts.items())),
        "activation_pricing_branches": dict(sorted(branch_counts.items())),
        "by_format": {
            fmt: row for fmt, row in sorted(by_format.items())
        },
    }


def summarize_applicability_masks(records: list[dict]) -> dict:
    """Summarize format candidates removed before the optimizer sees them.

    The allocator's legality gate is part of the optimization layer: illegal
    candidates are excluded before DP, rather than caught later by export.
    This summary is intentionally small enough to save beside Pareto curves
    while still preserving exact qnames and kernel shapes for debugging.
    """
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_shape: dict[tuple[str, str], dict[tuple[int, int], dict]] = defaultdict(dict)
    for rec in records:
        fmt = str(rec.get("format", ""))
        reason = str(rec.get("reason", "not_applicable"))
        summary[fmt][reason] += 1
        out_features = int(rec.get("out_features", 0) or 0)
        in_features = int(rec.get("in_features", 0) or 0)
        shape_key = (out_features, in_features)
        bucket_key = (fmt, reason)
        bucket = by_shape[bucket_key].setdefault(shape_key, {
            "shape": [out_features, in_features],
            "count": 0,
            "sample": [],
            "detail": rec.get("detail", ""),
        })
        bucket["count"] += 1
        if len(bucket["sample"]) < 8:
            bucket["sample"].append(rec.get("qname", ""))

    shape_payload: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for (fmt, reason), shapes in by_shape.items():
        shape_payload[fmt][reason] = sorted(
            shapes.values(),
            key=lambda row: (-int(row["count"]), row["shape"]),
        )

    return {
        "summary": {
            fmt: dict(sorted(reason_counts.items()))
            for fmt, reason_counts in sorted(summary.items())
        },
        "by_shape": {
            fmt: {
                reason: rows
                for reason, rows in sorted(reason_map.items())
            }
            for fmt, reason_map in sorted(shape_payload.items())
        },
        "records": sorted(
            records,
            key=lambda row: (
                str(row.get("format", "")),
                str(row.get("reason", "")),
                int(row.get("out_features", 0) or 0),
                int(row.get("in_features", 0) or 0),
                str(row.get("qname", "")),
            ),
        ),
    }


def _member_activation_branch(
    member_candidates: dict[str, dict[str, Candidate]],
    members: list[str],
    fmt: str,
) -> str | None:
    """The activation-pricing branch of an AGGREGATED super item.

    A super item's Δloss is the SUM of its members', so a single member
    priced on an uncalibrated weight-only branch taints the whole unit's
    claim. Unanimity reports the branch; disagreement is reported as
    ``mixed:<a>+<b>`` rather than collapsed to the majority, because "some
    rows of this serving unit never saw an activation measurement" is
    precisely the fact the stamp exists to preserve.
    """
    branches = sorted({
        str(member_candidates[m][fmt].activation_pricing)
        for m in members
        if member_candidates[m][fmt].activation_pricing is not None
    })
    if not branches:
        return None
    if len(branches) == 1:
        return branches[0]
    return "mixed:" + "+".join(branches)


def _member_serving_lane(
    member_candidates: dict[str, dict[str, Candidate]],
    members: list[str],
    fmt: str,
) -> object | None:
    """The serving-lane route of an aggregated super item.

    Every member of a fused-sibling / packed serving group loads under ONE
    format, and the lane is a function of the format and the target profile,
    so the members' routes are identical by construction.
    """
    for m in members:
        lane = member_candidates[m][fmt].serving_lane
        if lane is not None:
            return lane
    return None


_FUSED_SIBLING_MARKER = ".__siblings__."


def aggregate_fused_siblings(
    stats: dict,
    costs: dict,
    formats: list[fr.FormatSpec],
    candidates: dict[str, list[Candidate]],
    profile,
    calibrated_gains: dict[str, float] | None = None,
    activation_pricing: ActivationFairPricing | None = None,
) -> tuple[dict, dict, dict]:
    """Aggregate fused siblings into single DP items.

    A group whose members share NO legal format is a HARD ERROR, not a
    fallback to individual rows. Fused siblings (q/k/v, gate/up) must load
    under ONE format — that is a serving invariant, not a preference — so
    members with disjoint menus cannot be coherently promoted at all: whatever
    format whole-group promotion lands on is illegal for at least one member,
    and ``compute_achieved`` now refuses to price that state (AssertionError,
    at every target) rather than scoring the unpriced member at zero Δloss.
    Individual rows would therefore only defer the same failure past the solve,
    while a silently missing ``candidates_ext`` entry (the pre-fix behavior:
    ``stats_ext``/``costs_ext`` assigned, ``candidates_ext`` not) drops the
    whole group from the DP with no error at all. The same argument is written
    out at length in ``aggregate_packed_serving_groups``.
    """
    if profile is None:
        return stats, costs, candidates

    gains = calibrated_gains or {}
    ucb_z = _cost_ucb_z()
    grouped: dict[str, list[str]] = {}
    ungrouped: list[str] = []
    for name in candidates:
        if _FUSED_SIBLING_MARKER in name or _PACKED_GROUP_MARKER in name:
            ungrouped.append(name)
            continue
        try:
            key = profile.fused_sibling_group(name)
        except Exception:
            key = None
        if key is None:
            ungrouped.append(name)
            continue
        grouped.setdefault(key, []).append(name)

    for key in list(grouped.keys()):
        if len(grouped[key]) < 2:
            ungrouped.extend(grouped.pop(key))

    if not grouped:
        return stats, costs, candidates

    stats_ext = {n: stats[n] for n in ungrouped}
    costs_ext = {n: costs.get(n, {}) for n in ungrouped}
    candidates_ext = {n: candidates[n] for n in ungrouped}

    for key, members in grouped.items():
        members = sorted(members)
        safe_key = key.replace(".", "__")
        super_name = f"{members[0].rsplit('.', 1)[0]}{_FUSED_SIBLING_MARKER}{safe_key}"

        n_params = sum(stats[m]["n_params"] for m in members)
        sum_h = sum(stats[m]["h_trace"] for m in members)
        d_out = int(stats[members[0]].get("out_features", 0) or 0)
        d_in = int(stats[members[0]].get("in_features", 0) or 0)

        stats_ext[super_name] = {
            "h_trace": sum_h,
            "h_trace_raw": sum(stats[m].get("h_trace_raw", 0.0) for m in members),
            "h_w2_sum": sum(stats[m].get("h_w2_sum", 0.0) for m in members),
            "w_max_abs": max(stats[m].get("w_max_abs", 0.0) for m in members),
            "w_norm_sq": sum(stats[m].get("w_norm_sq", 0.0) for m in members),
            "n_params": n_params,
            "in_features": d_in,
            "out_features": d_out,
            "n_tokens_seen": sum(stats[m].get("n_tokens_seen", 0) for m in members),
            "_fused_siblings": members,
            "_memory_bytes_by_format": {},
        }

        super_cost = {}
        super_cost_entry_fmt: dict[str, str] = {}
        for spec in formats:
            resolved_entries: list[tuple[str, dict]] = []
            missing = []
            for m in members:
                entry, entry_fmt = _resolve_cost_entry(
                    costs.get(m, {}), spec.name)
                if entry is None or "error" in entry:
                    missing.append(m)
                else:
                    resolved_entries.append((entry_fmt, entry))
            if missing:
                super_cost[spec.name] = {"error": "partial"}
                continue
            if resolved_entries:
                super_cost_entry_fmt[spec.name] = resolved_entries[0][0]
            sum_pred = 0.0
            member_terms = []
            # P5a: the per-family activation penalty is a MULTIPLIER on the
            # weight-only branch, so it scales the member dloss and the
            # member stderr identically — passed as the hedge's scale below
            # (the calibrated gain enters once, later).
            act_penalty = _activation_penalty(spec.name, activation_pricing)
            for m, (_entry_fmt, c) in zip(members, resolved_entries):
                # Mirrors build_candidates, including unmeasured packed
                # output_mse fallback, bit-exact short-circuit, and
                # format-alias lookup. The calibrated gain is applied ONCE to
                # the summed super-item dloss (below, at candidate build), so
                # the per-member terms here — and the hedge conversion — are
                # un-gained.
                member_pred = cost_entry_predicted_dloss(
                    stats[m], c, format_name=spec.name,
                    activation_pricing=activation_pricing)
                sum_pred += member_pred
                member_terms.append((stats[m], c, member_pred, act_penalty))
            # Same UCB conversion as aggregate_packed_serving_groups: the
            # LINEAR z·Σ(stderr) baked into sum_pred becomes the independence
            # z·sqrt(Σ stderr²) (a qkv triple over-hedged at 3x linear now
            # hedges at √3), and the aggregated stderr is stored so consumers
            # of the aggregated cost table keep the hedge instead of silently
            # reading stderr 0. weight_mse is derived from the UN-hedged sum
            # so the super entry's mse is not z-contaminated.
            hedge_linear, stderr_agg = _super_item_ucb_hedge(
                member_terms, ucb_z)
            base_pred = sum_pred - hedge_linear
            effective_mse = base_pred / (0.5 * sum_h) if sum_h > 0 else 0.0
            super_cost[spec.name] = {
                "weight_mse": effective_mse,
                "predicted_dloss": base_pred,
                "predicted_dloss_stderr": stderr_agg,
            }
            if activation_pricing is not None:
                # The members' penalties are already inside base_pred; mark
                # the super entry so a re-price cannot square the correction.
                super_cost[spec.name][APPLIED_MARKER_KEY] = True
        costs_ext[super_name] = super_cost

        member_format_sets = [
            {c.fmt for c in candidates.get(m, [])}
            for m in members
        ]
        if member_format_sets:
            member_format_intersection = set.intersection(*member_format_sets)
        else:
            member_format_intersection = set()
        if not member_format_intersection:
            raise AssertionError(
                _fused_group_menu_error(
                    super_name,
                    key,
                    members,
                    candidates,
                    member_format_intersection,
                    formats,
                    "share no legal format",
                )
            )

        member_by_name = {
            member: {
                candidate.fmt: candidate for candidate in candidates[member]
            }
            for member in members
        }
        cands = []
        for spec in formats:
            if spec.name not in member_format_intersection:
                continue
            entry = super_cost.get(spec.name)
            if entry is None or "error" in entry:
                continue
            total_bytes = sum(
                int(member_by_name[m][spec.name].memory_bytes) for m in members
            )
            serialized_identities = sorted({
                identity
                for m in members
                for identity in (member_by_name[m][spec.name].serialized_identity,)
                if identity is not None
            })
            serialized_sidecar_identities = sorted({
                identity
                for m in members
                for identity in (
                    member_by_name[m][spec.name].serialized_sidecar_identity,
                )
                if identity is not None
            })
            bits_per_param = 8.0 * total_bytes / max(n_params, 1)
            stats_ext[super_name]["_memory_bytes_by_format"][spec.name] = total_bytes
            entry_fmt = super_cost_entry_fmt.get(spec.name, spec.name)
            gain = float(gains.get(spec.name, gains.get(entry_fmt, 1.0)))
            # gain·(base + z·stderr_agg) == gain·base + z·sqrt(Σ (stderr·gain)²)
            # for the single group-wide gain this path applies, i.e. exactly the
            # packed path's construction. At z == 0 this is gain·sum_pred,
            # bit-for-bit what this path produced before the hedge fix.
            predicted = (
                entry["predicted_dloss"]
                + ucb_z * float(entry.get("predicted_dloss_stderr", 0.0))
            ) * gain
            cands.append(Candidate(
                fmt=spec.name,
                bits_per_param=bits_per_param,
                memory_bytes=total_bytes,
                predicted_dloss=max(predicted, 0.0),
                activation_pricing=_member_activation_branch(
                    member_by_name, members, spec.name),
                serving_lane=_member_serving_lane(
                    member_by_name, members, spec.name),
                serialized_identity=(
                    json.dumps(serialized_identities, separators=(",", ":"))
                    if serialized_identities else None
                ),
                serialized_sidecar_identity=(
                    json.dumps(
                        serialized_sidecar_identities,
                        separators=(",", ":"),
                    )
                    if serialized_sidecar_identities else None
                ),
            ))
        if not cands:
            # Unreachable via the intersection (a common candidate format
            # implies a non-error cost row for every member), so this catches
            # the residual case: an aggregation menu that does not contain the
            # formats the member candidates were built from. Same verdict —
            # stats_ext/costs_ext are already written, so returning here would
            # drop the group from the DP silently.
            raise AssertionError(
                _fused_group_menu_error(
                    super_name,
                    key,
                    members,
                    candidates,
                    member_format_intersection,
                    formats,
                    "share legal formats that this aggregation menu does not "
                    "price",
                )
            )
        candidates_ext[super_name] = cands

    return stats_ext, costs_ext, candidates_ext


def _fused_group_menu_error(
    super_name: str,
    key: str,
    members: list[str],
    candidates: dict[str, list[Candidate]],
    intersection: set[str],
    formats: list[fr.FormatSpec],
    what: str,
) -> str:
    """Diagnostic for a fused-sibling group that cannot be given one format."""
    member_lines = "\n".join(
        f"    {m}: legal={sorted(c.fmt for c in candidates.get(m, []))}"
        for m in members
    )
    return (
        f"fused-sibling group {key!r} (DP unit {super_name!r}) has "
        f"{len(members)} members that {what}:\n"
        f"{member_lines}\n"
        f"    common formats: {sorted(intersection)}\n"
        f"    aggregation menu: {[s.name for s in formats]}\n"
        "Fused siblings (q/k/v, gate/up) MUST load under one format, so an "
        "empty intersection is not an allocatable state: any format whole-"
        "group promotion picks is illegal for at least one member, and "
        "compute_achieved refuses to price that (it would otherwise score the "
        "unpriced member at zero Δloss and bias the min-Δloss ratchet toward "
        "exactly this state). Falling back to individual rows would only defer "
        "the same failure past the solve. This is an upstream cost/legality "
        "bug to fix, not a state to allocate around: a missing cost row for "
        "one sibling, an over-tight applicability mask (see the "
        "[alloc] format-applicability log lines and the mask summary JSON), or "
        "a passthrough-source mismatch (BF16/FP8_SOURCE are legal only where "
        "the source tensor already has that precision, so a group whose "
        "members have different source dtypes loses them)."
    )


def expand_fused_sibling_assignment(assignment: dict[str, str],
                                    stats_ext: dict) -> dict[str, str]:
    """Broadcast a fused-sibling super-item assignment back to members."""
    out = {}
    for name, fmt in assignment.items():
        if _FUSED_SIBLING_MARKER in name:
            members = stats_ext[name].get("_fused_siblings", [])
            for m in members:
                out[m] = fmt
        else:
            out[name] = fmt
    return out


_PACKED_GROUP_MARKER = ".__packed_serving__."


def aggregate_packed_serving_groups(
    stats: dict,
    costs: dict,
    formats: list[fr.FormatSpec],
    candidates: dict[str, list[Candidate]],
    profile,
    calibrated_gains: dict[str, float] | None = None,
    activation_pricing: ActivationFairPricing | None = None,
) -> tuple[dict, dict, dict]:
    """Aggregate packed-MoE serving groups into single DP decision units.

    A packed serving group (``profile.packed_expert_format_group``) is
    atomic at serve time: vLLM's FusedMoE loads every projection of every
    routed expert in a layer under ONE quantization scheme, so a "one row
    upgraded" DP decision is not a real option — the serving constraint
    charges the whole group. Pricing upgrades per row inside the DP while
    ``promote_serving_units`` charges the whole group is a ~1000x price
    mismatch: mispriced expert rows
    top the per-bin ranking, the feasibility tightening over-corrects, and
    cheap-to-upgrade dense rows starve while headroom goes unused.

    This pre-pass makes each packed group ONE multi-choice DP item whose
    per-format cost is the exact sum of member predicted_dloss and whose
    byte cost is the exact sum of member bytes at that format — so the DP
    and the serving constraint price identical moves and post-DP MoE
    promotion becomes a validated no-op. Only formats legal for EVERY
    member are offered (member candidate sets already encode source /
    profile / kernel-shape applicability).

    A group with NO common legal format falls back to individual rows, which
    keeps it visible and attributable instead of silently vanishing from the
    DP — but that state is NOT allocatable and the fallback does NOT "repair
    coherence". Members can only be assigned from their own (by definition
    disjoint) candidate lists, so whole-group promotion necessarily lands on a
    format that is illegal for at least one member, and ``compute_achieved``
    refuses to price it (AssertionError, at every target) rather than scoring
    the unpriced member at zero Δloss — which would make the illegal state
    look CHEAPEST to the min-Δloss ratchet. An empty intersection is an
    upstream cost/legality bug to fix (a missing cost row, an over-tight
    applicability mask, a passthrough-source mismatch), not a state to
    allocate around.

    Non-grouped rows (attention, shared/dense MLP) pass through untouched.
    Extrapolated expert cost rows are ordinary members. Use
    ``expand_packed_group_assignment`` to broadcast a group decision back
    to per-tensor entries for emission.
    """
    group_fn = getattr(profile, "packed_expert_format_group", None) \
        if profile is not None else None
    if not callable(group_fn):
        return stats, costs, candidates

    gains = calibrated_gains or {}
    ucb_z = _cost_ucb_z()
    grouped: dict[str, list[str]] = {}
    ungrouped: list[str] = []
    for name in candidates:
        if _FUSED_SIBLING_MARKER in name or _PACKED_GROUP_MARKER in name:
            ungrouped.append(name)
            continue
        try:
            key = group_fn(name)
        except PackedExpertRoleUnknown:
            # An explicit "the profile cannot describe this unit" verdict must
            # not be swallowed into "this row has no group" (see
            # PackedExpertRoleUnknown).
            raise
        except Exception:
            key = None
        if key is None:
            ungrouped.append(name)
            continue
        grouped.setdefault(key, []).append(name)

    for key in list(grouped.keys()):
        if len(grouped[key]) < 2:
            ungrouped.extend(grouped.pop(key))

    if not grouped:
        return stats, costs, candidates

    stats_ext = {n: stats[n] for n in ungrouped}
    costs_ext = {n: costs.get(n, {}) for n in ungrouped}
    candidates_ext = {n: candidates[n] for n in ungrouped}

    for key, members in sorted(grouped.items()):
        members = sorted(members)
        safe_key = key.replace(".", "__")
        super_name = (
            f"{members[0].rsplit('.', 1)[0]}{_PACKED_GROUP_MARKER}{safe_key}"
        )
        member_cands = {
            m: {c.fmt: c for c in candidates[m]} for m in members
        }
        common_fmts = set.intersection(
            *(set(per_member) for per_member in member_cands.values())
        )
        n_params = sum(int(stats[m]["n_params"]) for m in members)
        memory_by_fmt: dict[str, int] = {}
        super_cost: dict[str, dict] = {}
        cands: list[Candidate] = []
        for spec in formats:
            if spec.name not in common_fmts:
                continue
            total_bytes = sum(
                int(member_cands[m][spec.name].memory_bytes) for m in members
            )
            sum_pred = sum(
                float(member_cands[m][spec.name].predicted_dloss)
                for m in members
            )
            # UCB hedge (PRISMAQUANT_COST_UCB_Z > 0): each member candidate
            # was priced independently, so the sum above carries a LINEAR
            # z·Σ(stderr·gain) hedge. Convert it to the independence
            # aggregate and store the aggregated stderr on the super cost
            # entry so consumers of the aggregated cost table keep the hedge
            # instead of silently reading stderr 0. At z == 0 this is a no-op
            # and the per-format dloss stays the exact sum of member
            # candidates. Shared with aggregate_fused_siblings.
            member_terms = []
            # P5a: member candidates were priced WITH the family penalty, so
            # the hedge conversion must scale each member's stderr by the same
            # factor or it would over-subtract the linear hedge it is undoing.
            act_penalty = _activation_penalty(spec.name, activation_pricing)
            for m in members:
                entry, entry_fmt = _resolve_cost_entry(
                    costs.get(m, {}), spec.name)
                member_terms.append((
                    stats[m],
                    entry,
                    float(member_cands[m][spec.name].predicted_dloss),
                    float(gains.get(spec.name, gains.get(entry_fmt, 1.0)))
                    * act_penalty,
                ))
            hedge_linear, stderr_agg = _super_item_ucb_hedge(
                member_terms, ucb_z)
            base_pred = sum_pred - hedge_linear
            hedged_pred = base_pred + ucb_z * stderr_agg
            memory_by_fmt[spec.name] = total_bytes
            super_cost[spec.name] = {
                "predicted_dloss": base_pred,
                "predicted_dloss_stderr": stderr_agg,
            }
            if activation_pricing is not None:
                super_cost[spec.name][APPLIED_MARKER_KEY] = True
            cands.append(Candidate(
                fmt=spec.name,
                bits_per_param=8.0 * total_bytes / max(n_params, 1),
                memory_bytes=total_bytes,
                predicted_dloss=max(hedged_pred, 0.0),
                activation_pricing=_member_activation_branch(
                    member_cands, members, spec.name),
                serving_lane=_member_serving_lane(
                    member_cands, members, spec.name),
                serialized_identity=(
                    json.dumps(sorted({
                        identity
                        for m in members
                        for identity in (
                            member_cands[m][spec.name].serialized_identity,
                        )
                        if identity is not None
                    }), separators=(",", ":"))
                    if any(
                        member_cands[m][spec.name].serialized_identity is not None
                        for m in members
                    ) else None
                ),
                serialized_sidecar_identity=(
                    json.dumps(sorted({
                        identity
                        for m in members
                        for identity in (
                            member_cands[m][spec.name]
                            .serialized_sidecar_identity,
                        )
                        if identity is not None
                    }), separators=(",", ":"))
                    if any(
                        member_cands[m][spec.name]
                        .serialized_sidecar_identity is not None
                        for m in members
                    ) else None
                ),
            ))
        if not cands:
            # No format is legal for every member; aggregating would drop the
            # whole group from the DP. Keep the members as individual rows
            # (pre-refactor behavior) so the group stays visible and the
            # failure is attributable — NOT because promotion can repair it.
            # It cannot: the members' candidate lists are disjoint here, so
            # promotion always lands on a format illegal for some member and
            # every solve at every target ends in a compute_achieved pricing
            # error. See this function's docstring.
            for m in members:
                stats_ext[m] = stats[m]
                costs_ext[m] = costs.get(m, {})
                candidates_ext[m] = candidates[m]
            continue
        # NOTE: deliberately NO in_features/out_features here. A packed
        # group mixes member shapes (gate/up vs down projections), so no
        # single (out, in) pair describes it — copying members[0]'s shape
        # would make _shape_from_stats compute ONE member's bytes for the
        # whole group. Without them _shape_from_stats falls back to the
        # rank-1 (n_params,) legacy shape, which is at least
        # total-parameter-consistent; exact byte paths must (and do)
        # prefer the candidate / _memory_bytes_by_format.
        stats_ext[super_name] = {
            "h_trace": sum(
                float(stats[m].get("h_trace", 0.0) or 0.0) for m in members
            ),
            "n_params": n_params,
            "n_tokens_seen": sum(
                int(stats[m].get("n_tokens_seen", 0) or 0) for m in members
            ),
            "_packed_group_members": members,
            "_packed_group_key": key,
            "_memory_bytes_by_format": memory_by_fmt,
        }
        costs_ext[super_name] = super_cost
        candidates_ext[super_name] = cands

    return stats_ext, costs_ext, candidates_ext


def expand_packed_group_assignment(assignment: dict[str, str],
                                   stats_ext: dict) -> dict[str, str]:
    """Broadcast a packed-serving-group decision back to member tensors."""
    out = {}
    for name, fmt in assignment.items():
        if _PACKED_GROUP_MARKER in name:
            members = stats_ext[name].get("_packed_group_members", [])
            for m in members:
                out[m] = fmt
        else:
            out[name] = fmt
    return out


class _RoleSplitProfile:
    """Profile view that splits packed serving groups by projection role.

    Wraps a model profile so ``packed_expert_format_group`` returns a
    (layer, role-group) key — gate+up projections form one serving unit and
    down projections another (2 units per MoE layer instead of 1). Because
    BOTH the DP aggregation and ``promote_serving_units`` key groups through
    the profile, wrapping keeps them consistent: role units stay atomic,
    and the final serving promotion remains a validated no-op. Everything
    else delegates to the wrapped profile.

    The role itself comes from ``profile.packed_expert_role_group``: expert
    leaf naming (``gate_proj``/``up_proj`` vs LFM2.5's ``w1``/``w3``, and which
    packed 3D parent each belongs to) is profile knowledge, and the allocator
    does not parse model names — the same boundary
    ``packed_expert_format_group`` already respects.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def packed_expert_format_group(self, qname: str) -> str | None:
        key = self._inner.packed_expert_format_group(qname)
        if key is None:
            return None
        role_fn = getattr(self._inner, "packed_expert_role_group", None)
        if not callable(role_fn):
            raise PackedExpertRoleUnknown(
                f"profile {type(self._inner).__name__} groups {qname!r} as a "
                "packed-expert serving unit but has no "
                "packed_expert_role_group accessor, so the requested "
                "gate_up/down role split cannot be keyed. Implement it (or "
                "inherit ModelProfile, which derives it from the profile's "
                "packed_experts spec)."
            )
        role = role_fn(qname)
        if role is None:
            raise PackedExpertRoleUnknown(
                f"profile {type(self._inner).__name__} groups {qname!r} as a "
                f"packed-expert serving unit (key {key!r}) but declares no "
                "role for its projection leaf, so the requested gate_up/down "
                "split would silently degrade to one layer-uniform unit. "
                "Declare the leaf in the profile's structure spec "
                "(packed_experts.projection_splits maps each per-expert leaf "
                "to its packed 3D parent, e.g. "
                '{"gate_up_proj": ["w1", "w3"], "down_proj": ["w2"]}).'
            )
        return f"{key}::role:{role}"


def packed_role_split_profile(profile):
    """Wrap ``profile`` so packed expert groups split into gate_up / down
    serving units. Pass-through when the profile has no packed groups."""
    if profile is None or not callable(
            getattr(profile, "packed_expert_format_group", None)):
        return profile
    return _RoleSplitProfile(profile)


def _scan_source_dtype_manifest(
    model_path: str,
    profile=None,
) -> dict[str, str]:
    """Classify source Linear weights for passthrough gating.

    Returns ``bf16`` only for actual BF16 source tensors, ``fp8`` for native
    FP8/scale-sidecar sources, and ``other`` for FP16/FP32/etc. passthroughs
    that would be synthesized rather than byte-preserving.
    """
    from safetensors import safe_open

    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    weight_map = {}
    if idx_path.exists():
        try:
            with open(idx_path) as f:
                weight_map = json.load(f).get("weight_map", {})
        except Exception:
            weight_map = {}
    if not weight_map:
        for shard in sorted(src.glob("*.safetensors")):
            try:
                with safe_open(str(shard), framework="pt", device="cpu") as sf:
                    for key in sf.keys():
                        weight_map.setdefault(key, shard.name)
            except Exception:
                continue
    # Packed-MoE expert params are checkpoint keys with NO ``.weight``
    # suffix (LFM2.5 packed, Qwen3.6-35B: ``...experts.gate_up_proj``).
    # Without classifying them the manifest has no source kind for the
    # packed recipe names the allocator costs, and the BF16 passthrough is
    # dropped (source_dtype_mismatch) on a BF16 source — an expert-menu
    # completeness bug. (Per-expert INDEXED layouts store 2-D ``.weight``
    # keys and classify their own recipe names via the normal path.)
    import re as _re
    _packed_leaf_re = _re.compile(
        r"\.experts\.(?:gate_up_proj|down_proj|gate_proj|up_proj|w1|w2|w3)$"
    )
    bases: dict[str, set[str]] = {}
    packed_bases: set[str] = set()
    for key in weight_map:
        matched = False
        # ``.scale`` is the OCP-MX / DSv4 group-scale sibling spelling, and it
        # is listed LAST so it can never shadow ``.weight_scale``: the two do
        # not overlap as suffixes ("...weight_scale"[-6:] == "_scale", not
        # ".scale"), but ordering makes that independent of that coincidence.
        for suffix in (".weight_scale_inv", ".weight_scale", ".weight",
                       ".scale"):
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                bases.setdefault(base, set()).add(suffix[1:])
                matched = True
                break
        if not matched and _packed_leaf_re.search(key):
            bases.setdefault(key, set()).add("weight")
            packed_bases.add(key)
    weight_dtypes: dict[str, str] = {}
    scale_dtypes: dict[str, str] = {}
    shard_keys: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        if not (key.endswith(".weight") or key.endswith(".scale")
                or key in packed_bases):
            continue
        path = src / str(shard)
        if path.is_file():
            shard_keys[str(path)].append(key)
    for path, keys in shard_keys.items():
        try:
            with safe_open(path, framework="pt", device="cpu") as sf:
                for key in keys:
                    is_scale = key.endswith(".scale")
                    if is_scale:
                        base = key[: -len(".scale")]
                    elif key.endswith(".weight"):
                        base = key[: -len(".weight")]
                    else:
                        base = key
                    try:
                        dtype = str(sf.get_slice(key).get_dtype()).upper()
                    except Exception:
                        continue
                    if is_scale:
                        scale_dtypes[base] = dtype
                    else:
                        weight_dtypes[base] = dtype
        except Exception:
            continue

    def _strip_weight_suffix(name: str) -> str:
        return name[:-7] if name.endswith(".weight") else name

    def _to_recipe_name(ck_base: str) -> str:
        if ck_base.startswith("mtp."):
            # MTP tensors are REAL source tensors stored under the recipe
            # namespace itself (transformers v5 drops the module; prismaquant
            # synthesizes it back under the same names, and probe/cost rows
            # use them verbatim). The historical skip left MTP names with no
            # source kind, so the BF16 passthrough was dropped
            # (source_dtype_mismatch) and --mtp-format=BF16 hard-failed the
            # moment MTP rows were actually costed (35B frontier, 2026-07-02).
            return ck_base
        weight_key = f"{ck_base}.weight"
        if profile is not None:
            mapper = getattr(profile, "checkpoint_to_live_name", None)
            if callable(mapper):
                try:
                    live_param = mapper(weight_key, multimodal=False)
                except TypeError:
                    live_param = mapper(weight_key)
                except Exception:
                    live_param = None
                if live_param is None:
                    return ""
                live_qname = _strip_weight_suffix(str(live_param))
                recipe_mapper = getattr(profile, "live_to_recipe_name", None)
                if callable(recipe_mapper):
                    try:
                        return str(recipe_mapper(live_qname))
                    except Exception:
                        return live_qname
                return live_qname
        if (ck_base.startswith("model.visual.")
                or ck_base.startswith("model.audio_tower.")
                or ck_base.startswith("model.vision_tower.")
                or ck_base.startswith("model.embed_vision.")
                or ck_base.startswith("model.embed_audio.")):
            return ""
        if ck_base.startswith("model.language_model."):
            return "model." + ck_base[len("model.language_model."):]
        return ck_base

    def _packed_to_recipe_name(ck_key: str) -> str:
        # Packed expert params have no ``.weight`` to fabricate for
        # checkpoint_to_live_name; checkpoint name == live name modulo the
        # language_model prefix, then the profile's live->recipe mapping.
        name = ck_key
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model."):]
        if profile is not None:
            recipe_mapper = getattr(profile, "live_to_recipe_name", None)
            if callable(recipe_mapper):
                try:
                    return str(recipe_mapper(name))
                except Exception:
                    return name
        return name

    manifest: dict[str, str] = {}
    for base, suffixes in bases.items():
        if "weight" not in suffixes:
            continue
        dtype = weight_dtypes.get(base)
        scale_dtype = scale_dtypes.get(base)
        # E8M0 group/block exponents are the discriminator for the two native
        # formats FP8_SOURCE cannot represent. Both tests run BEFORE the
        # generic ``F8`` test, because the SCALE plane of both is itself
        # ``F8_E8M0`` and would otherwise be mistaken for an fp8 weight
        # contract with an FP32 scale_inv plane — a format whose byte count
        # and whose exported scale dtype are both wrong for this checkpoint.
        if scale_dtype == "F8_E8M0" and dtype in {"I8", "U8"}:
            # Nibble-packed 4-bit elements carried in an 8-bit container with
            # a power-of-two group scale: OCP-MX MXFP4 as DeepSeek-V4 ships
            # its routed experts.
            source_kind = "mxfp4"
        elif scale_dtype == "F8_E8M0" and dtype == "F8_E4M3":
            # Block-FP8 with UE8M0 block exponents (DeepSeek-V3.1/V4), NOT
            # the FP32 weight_scale_inv contract FP8_SOURCE models.
            source_kind = "fp8_ue8m0"
        elif "weight_scale_inv" in suffixes or "weight_scale" in suffixes:
            source_kind = "fp8"
        elif dtype == "BF16":
            source_kind = "bf16"
        elif dtype is not None and dtype.startswith("F8"):
            source_kind = "fp8"
        elif dtype is None:
            source_kind = "bf16"
        else:
            source_kind = "other"
        recipe_name = (
            _packed_to_recipe_name(base) if base in packed_bases
            else _to_recipe_name(base)
        )
        if not recipe_name:
            continue
        manifest[recipe_name] = source_kind
    fp8_pairs = None
    if profile is not None:
        pairs_fn = getattr(profile, "fp8_scale_pairs", None)
        if callable(pairs_fn):
            try:
                fp8_pairs = pairs_fn(model_path)
            except Exception:
                fp8_pairs = None
    if fp8_pairs:
        recipe_mapper = getattr(profile, "live_to_recipe_name", None)
        for live_param in fp8_pairs:
            live_qname = _strip_weight_suffix(str(live_param))
            if callable(recipe_mapper):
                try:
                    live_qname = str(recipe_mapper(live_qname))
                except Exception:
                    pass
            # A profile's ``fp8_scale_pairs`` answers "does this weight have a
            # serialized scale sibling the dequant pass must read", which is
            # true of EVERY block-scaled format — DSv4's MXFP4 experts and its
            # UE8M0 body both appear there. It is not a claim about which fp8
            # contract the bytes are, so it must not overwrite a kind the
            # header scan derived from the actual dtypes: doing so is what
            # made 33,024 packed-MXFP4 experts look like FP32-scaled fp8.
            # It still fills in names the dtype scan could not classify.
            if manifest.get(live_qname) in (None, "other"):
                manifest[live_qname] = "fp8"
    return manifest
