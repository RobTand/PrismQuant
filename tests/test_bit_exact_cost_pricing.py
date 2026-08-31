"""Bit-exact pricing: weight_mse == 0.0 short-circuits to zero dloss ONLY
for formats whose activation path is the identity.

Regression #1 (the original motivation): MX re-encodes of QAT/FP8 sources
store the source weights verbatim (``weight_mse == 0.0`` exactly), yet the
cost pipeline records a positive ``output_mse`` for the weight-only
passthrough formats too (kernel dequant dtype noise), which inverted
dominance against lossy k-quants.

Regression #2 (the review catch): the short-circuit must NOT fire for
W·A· formats. ``measure_quant_cost`` applies
``activation_quantize_dequantize(X)`` before computing ``output_mse``, so
for a weight-lossless activation-quantizing format (MXFP4 re-encode of an
MXFP4-packed source, MXFP8_E4M3 of an FP8-block source, ...) that
output_mse is REAL A-side error. Pricing those entries at dloss 0.0 makes
them the unbeatable global minimum at any budget while the served
activations are still quantized.

Contract pinned here:
  - weight-bit-exact + PASSTHROUGH-activation format (act_bits None or >= 16)
    short-circuits to dloss 0.0 ("bit_exact" source);
  - weight-bit-exact + ACTIVATION-QUANTIZING format keeps its measured
    output_mse pricing — at entry, build_candidates, and packed-group
    aggregation level;
  - unknown formats and explicit-cost_source entries never short-circuit;
  - the dtype-level fact used for the gate (FormatSpec.act_quant_changes_input
    <=> a non-identity activation_quantize_dequantize) holds for every
    registered format.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    build_candidates,
    cost_entry_is_bit_exact,
    cost_entry_predicted_dloss,
    cost_entry_source,
    cost_entry_uses_measured_output_mse,
)
from prismaquant.allocator_solver import solve_with_promotion


_STATS = {
    "h_trace": 2.5, "n_params": 1024 * 1024,
    "in_features": 1024, "out_features": 1024,
}

# A weight-only re-encode format: activations pass through unquantized.
_PASSTHROUGH_FMT = "MXFP8A16"
# A W·A· format: serving quantizes the input activations to FP4.
_ACTQUANT_FMT = "MXFP4"


def _bit_exact_entry():
    # Lossless weights; output probe carries A-side error and/or noise.
    return {"weight_mse": 0.0, "output_mse": 6.3e-3,
            "rel_output_mse": 8.7e-3}


def _lossy_entry():
    return {"weight_mse": 2.0e-6, "output_mse": 1.4e-3,
            "rel_output_mse": 1.9e-3}


def _random_activations():
    torch.manual_seed(1234)
    return torch.randn(4, 512, dtype=torch.float32) * 3.1


def test_registry_act_bits_declaration_matches_activation_callable():
    """act_quant_changes_input <=> activation_quantize_dequantize is NOT the
    identity, for every registered format.

    This is the dtype-level fact the bit-exact short-circuit relies on; pin it
    against the actual callables so the registry cannot drift. Note the
    equivalence is stated over the PROPERTY, not over ``act_bits is None``
    directly: a weight-only format may declare either ``None`` or ``>= 16``
    (see FormatSpec.act_quant_changes_input), and both are passthrough.
    """
    x = _random_activations()
    for name, spec in sorted(fr.REGISTRY.items()):
        # WO-A: tcq_trellis W4A4 lane has no unary QDQ — the served path
        # needs a calibrated input_global_scale per fused sibling, so the
        # FormatSpec's unary signature cannot supply it and must fail closed
        # with the contract string rather than silently approximate.
        if spec.family == "tcq_trellis":
            assert spec.act_quant_changes_input
            with pytest.raises(NotImplementedError, match="e2m1_group16_ue4m3_static"):
                spec.activation_quantize_dequantize(x.clone())
            continue
        out = spec.activation_quantize_dequantize(x.clone())
        if not spec.act_quant_changes_input:
            assert torch.equal(out, x), (
                f"{name}: act_bits={spec.act_bits} declares a passthrough "
                "activation path, but the callable changed the input")
        else:
            assert not torch.equal(out, x), (
                f"{name}: act_bits={spec.act_bits} declares activation "
                "quantization, but the callable is the identity")


def test_act_bits_16_is_a_passthrough_declaration(monkeypatch):
    """An A16 rung declared the NATURAL way (``act_bits=16`` — the spelling
    every autoround_config dict in format_registry already uses for BF16 /
    FP8_SOURCE / the W*A16 variants) means "activations are 16-bit", i.e. not
    quantized away from the execution dtype. It must be classified as
    passthrough, keep the bit-exact short-circuit, and satisfy the
    registry-wide equivalence above — not be mispriced as a W-and-A format
    and not turn that test red."""
    spec = fr.FormatSpec(
        name="_TEST_W4A16_EXPLICIT16",
        weight_bits=4, group_size=16, scale_bits=8,
        scale_dtype_name="fp8_e4m3", weight_element_dtype="fp4_e2m1",
        act_bits=16, act_dtype_name="bfloat16",
        activation_quantize_dequantize=lambda t: t,
    )
    assert not spec.act_quant_changes_input, (
        "act_bits=16 means 16-bit (unquantized) activations")
    # Satisfies the registry-wide equivalence rule.
    x = _random_activations()
    assert torch.equal(spec.activation_quantize_dequantize(x.clone()), x)

    monkeypatch.setitem(fr.REGISTRY, spec.name, spec)
    entry = _bit_exact_entry()
    assert cost_entry_is_bit_exact(entry, spec.name)
    assert cost_entry_predicted_dloss(
        _STATS, entry, format_name=spec.name) == 0.0
    assert cost_entry_source(_STATS, entry, spec.name) == "bit_exact"

    # The same weight-lossless entry at the W-and-A spelling of the same
    # nominal format keeps its measured A-side cost.
    wa = fr.FormatSpec(
        name="_TEST_W4A4_EXPLICIT4",
        weight_bits=4, group_size=16, scale_bits=8,
        scale_dtype_name="fp8_e4m3", weight_element_dtype="fp4_e2m1",
        act_bits=4, act_dtype_name="fp4_e2m1",
        activation_quantize_dequantize=lambda t: t * 0.5,
    )
    assert wa.act_quant_changes_input
    monkeypatch.setitem(fr.REGISTRY, wa.name, wa)
    assert not cost_entry_is_bit_exact(entry, wa.name)


def test_activation_quant_predicate_has_one_definition():
    """One predicate, one place: no module may re-derive "does this format
    quantize the activations?" from ``act_bits``. Consumers that re-derived it
    as ``act_bits is not None and act_bits < 16`` disagreed with the
    allocator's gate, so a format's activation semantics could differ between
    pricing and emulation."""
    root = Path(fr.__file__).resolve().parent
    allowed = {
        # The definition itself.
        "format_registry.py",
        # Parses compressed-tensors CONFIG dicts (act_bits as serialized
        # metadata), not FormatSpec objects.
        "validation_harness.py",
    }
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name in allowed:
            continue
        for lineno, line in enumerate(
                path.read_text().splitlines(), start=1):
            if "act_bits" not in line:
                continue
            if "is not None" in line or "< 16" in line:
                offenders.append(
                    f"{path.relative_to(root)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "use FormatSpec.act_quant_changes_input instead of re-deriving the "
        "predicate from act_bits:\n" + "\n".join(offenders))


def test_activation_quant_assignment_uses_the_shared_predicate():
    """The KL validator's activation-quant set is one of the four consumers
    unified onto the property: passthrough formats (including the A16
    variants) must not be emulated as activation-quantizing."""
    from prismaquant.validate_assignments_kl import (
        _activation_quant_assignment,
    )

    got = _activation_quant_assignment({
        "wa_nvfp4": "NVFP4",
        "wa_gguf": "Q4_K",
        "wa_fp8": "FP8_DYNAMIC",
        "a16_bf16": "BF16",
        "a16_fp8_source": "FP8_SOURCE",
        "a16_mxfp8": "MXFP8A16",
        "a16_nvfp4": "NVFP4A16",
        "a16_int8": "INT8_W8A16",
    })
    assert set(got) == {"wa_nvfp4", "wa_gguf", "wa_fp8"}
    for qname, fmt in got.items():
        assert fr.get_format(fmt).act_quant_changes_input, (qname, fmt)


def test_passthrough_activation_bit_exact_prices_at_zero_dloss():
    entry = _bit_exact_entry()
    assert cost_entry_is_bit_exact(entry, _PASSTHROUGH_FMT)
    assert cost_entry_predicted_dloss(
        _STATS, entry, format_name=_PASSTHROUGH_FMT) == 0.0
    # gain multipliers cannot resurrect a cost that is zero by construction
    assert cost_entry_predicted_dloss(
        _STATS, entry, gain=3.7, format_name=_PASSTHROUGH_FMT) == 0.0
    assert cost_entry_source(_STATS, entry, _PASSTHROUGH_FMT) == "bit_exact"
    assert not cost_entry_uses_measured_output_mse(
        _STATS, entry, _PASSTHROUGH_FMT)


def test_activation_quantizing_bit_exact_keeps_measured_a_side_cost():
    """weight_mse == 0.0 on a W·A· format proves nothing about the output:
    measure_quant_cost quantized the activations before measuring
    output_mse, so the measured cost is real and must be charged."""
    entry = _bit_exact_entry()
    assert not cost_entry_is_bit_exact(entry, _ACTQUANT_FMT)
    assert cost_entry_uses_measured_output_mse(_STATS, entry, _ACTQUANT_FMT)
    assert cost_entry_source(_STATS, entry, _ACTQUANT_FMT) == "output_mse"
    expected = 0.5 * _STATS["h_trace"] * entry["output_mse"]
    got = cost_entry_predicted_dloss(_STATS, entry, format_name=_ACTQUANT_FMT)
    assert abs(got - expected) < 1e-12
    # Every registered activation-quantizing format is gated the same way.
    for name, spec in sorted(fr.REGISTRY.items()):
        if spec.act_quant_changes_input:
            assert not cost_entry_is_bit_exact(_bit_exact_entry(), name), name


def test_unknown_format_never_short_circuits():
    """No format identity, no proof: the caller that cannot name the
    format keeps the conservative (pre-shortcut) pricing."""
    entry = _bit_exact_entry()
    assert not cost_entry_is_bit_exact(entry)
    assert not cost_entry_is_bit_exact(entry, None)
    assert not cost_entry_is_bit_exact(entry, "NOT_A_FORMAT")
    assert cost_entry_uses_measured_output_mse(_STATS, entry)
    expected = 0.5 * _STATS["h_trace"] * entry["output_mse"]
    assert abs(cost_entry_predicted_dloss(_STATS, entry) - expected) < 1e-12


def test_explicit_cost_source_entries_are_never_bit_exact():
    """Entries with an explicit cost_source (e.g. the production-render
    score pipeline) default weight_mse to 0.0 as a placeholder, not a
    measurement — the short-circuit must not override their own pricing,
    matching the precedence cost_entry_source gives the explicit source."""
    entry = {"predicted_dloss": 12.0, "weight_mse": 0.0, "output_mse": 0.0,
             "output_mse_measured": False,
             "cost_source": "production_render_score"}
    assert not cost_entry_is_bit_exact(entry, _PASSTHROUGH_FMT)
    assert cost_entry_predicted_dloss(
        _STATS, entry, format_name=_PASSTHROUGH_FMT) == 12.0
    assert cost_entry_source(
        _STATS, entry, _PASSTHROUGH_FMT) == "production_render_score"


def test_lossy_entries_keep_measured_output_mse_pricing():
    entry = _lossy_entry()
    assert not cost_entry_is_bit_exact(entry, _PASSTHROUGH_FMT)
    assert cost_entry_uses_measured_output_mse(_STATS, entry, _PASSTHROUGH_FMT)
    expected = 0.5 * _STATS["h_trace"] * entry["output_mse"]
    assert abs(
        cost_entry_predicted_dloss(
            _STATS, entry, format_name=_PASSTHROUGH_FMT
        ) - expected
    ) < 1e-12
    # Near-zero is not zero: only an exact 0.0 proves losslessness.
    tiny = dict(entry, weight_mse=1e-300)
    assert not cost_entry_is_bit_exact(tiny, _PASSTHROUGH_FMT)


def test_build_candidates_splits_passthrough_and_actquant_bit_exact():
    stats = {"row": dict(_STATS)}
    costs = {"row": {_PASSTHROUGH_FMT: _bit_exact_entry(),
                     _ACTQUANT_FMT: _bit_exact_entry(),
                     "Q4_K": _lossy_entry()}}
    specs = [fr.get_format(_PASSTHROUGH_FMT), fr.get_format(_ACTQUANT_FMT),
             fr.get_format("Q4_K")]
    cands = build_candidates(stats, costs, specs)
    by_fmt = {c.fmt: c for c in cands["row"]}
    assert by_fmt[_PASSTHROUGH_FMT].predicted_dloss == 0.0
    expected_a_side = 0.5 * _STATS["h_trace"] * _bit_exact_entry()["output_mse"]
    assert abs(by_fmt[_ACTQUANT_FMT].predicted_dloss - expected_a_side) < 1e-12
    assert by_fmt["Q4_K"].predicted_dloss > 0.0
    # The A-side cost keeps MXFP4 comparable with the lossy k-quant on the
    # SAME footing (both model their compute-path activation error) instead
    # of an unbeatable 0.0.
    assert by_fmt[_ACTQUANT_FMT].predicted_dloss > by_fmt["Q4_K"].predicted_dloss


def test_zero_dloss_fewer_bits_strictly_dominates_in_the_dp():
    """A measured-zero-dloss candidate at FEWER bits must displace a
    positive-dloss candidate at MORE bits whenever the DP funds an
    upgrade — dloss == 0 is a valid, optimal, measured cost."""
    from prismaquant.allocator_solver import Candidate

    stats = {}
    cands = {}
    menu = ["IQ2_XXS", "MXFP4", "Q4_K"]
    dloss = {"IQ2_XXS": 50.0, "MXFP4": 0.0, "Q4_K": 1.7}
    for i in range(4):
        name = f"model.layers.{i}.self_attn.o_proj"
        n = 1 << 20
        stats[name] = {"h_trace": 1.0, "n_params": n,
                       "in_features": 1024, "out_features": 1024}
        cands[name] = []
        for f in menu:
            bpp = fr.get_format(f).effective_bits
            cands[name].append(Candidate(
                fmt=f, bits_per_param=bpp,
                memory_bytes=int(round(bpp * n / 8.0)),
                predicted_dloss=dloss[f]))
    specs = {f: fr.get_format(f) for f in menu}
    rank = {f: i for i, f in enumerate(menu)}

    for target in (4.3, 4.6, 16.0):  # fits MXFP4 only / both / everything
        assign, achieved = solve_with_promotion(
            stats, cands, target, specs, rank, bit_precision=0.001)
        assert assign is not None
        assert achieved <= target + 0.01
        assert all(f == "MXFP4" for f in assign.values()), (
            f"target={target}: zero-dloss fewer-bits format must win, "
            f"got {assign}")


def _packed_stats(h_trace: float = 2.5):
    """A packed-expert row (what `_stats_indicates_packed_expert` recognizes).

    Shape/param counts are kept self-consistent with the 3D stacked tensor
    (`_shape_from_stats` returns ``(E, out, in)`` here) so byte math is real.
    """
    experts, out_features, in_features = 8, 1024, 1024
    return {
        "h_trace": h_trace,
        "n_params": experts * out_features * in_features,
        "in_features": in_features, "out_features": out_features,
        "_packed_experts_module": "model.layers.0.mlp.experts",
        "num_experts": experts,
    }


def _unmeasured_output_entry(weight_mse: float):
    """Exactly what measure_quant_cost writes when the packed routed forward
    could not be reconstructed (`can_measure_output` false) or the run set
    PRISMAQUANT_EXPERT_COST_SAMPLE: output_mse is a placeholder zero."""
    return {"weight_mse": weight_mse, "output_mse": 0.0, "rel_output_mse": 0.0,
            "output_mse_measured": False}


def test_unmeasured_activation_cost_is_excluded_not_priced_at_zero():
    """Regression #3: the zero-price pathology survives on rows where
    output_mse was never measured.

    ``cost_entry_is_bit_exact`` only closed the output_mse branch. With
    ``output_mse_measured=False`` pricing falls through to
    predicted_dloss/weight_mse, and a weight-lossless re-encode (MXFP4 over an
    MXFP4-packed source) is exactly 0.0 there too — the unbeatable global
    minimum at any budget for an assignment whose served activations are 4-bit.
    An unknown cost priced at the DP's optimum is always selected, so the
    candidate is excluded (counted + logged like any inapplicable format)
    instead of guessed."""
    from prismaquant.allocator_candidates import (
        ACTIVATION_COST_UNMEASURED_REASON,
        summarize_applicability_masks,
    )

    stats = {"e": _packed_stats()}
    costs = {"e": {
        _ACTQUANT_FMT: _unmeasured_output_entry(0.0),   # lossless re-encode
        "Q4_K": _unmeasured_output_entry(2.0e-6),       # lossy, still priced
        _PASSTHROUGH_FMT: _unmeasured_output_entry(0.0),
        "BF16": {"weight_mse": 0.0},
    }}
    specs = [fr.get_format(f) for f in
             (_ACTQUANT_FMT, "Q4_K", _PASSTHROUGH_FMT, "BF16")]
    records: list[dict] = []
    cands = build_candidates(stats, costs, specs, mask_records=records)

    by_fmt = {c.fmt: c for c in cands["e"]}
    assert _ACTQUANT_FMT not in by_fmt, (
        "a W-and-A format with an unmeasured activation path and zero weight "
        "error must not enter the DP at dloss 0.0")
    # Weight-lossless PASSTHROUGH-activation formats are unaffected: their
    # zero is proof of end-to-end losslessness, not an unmeasured unknown.
    assert by_fmt[_PASSTHROUGH_FMT].predicted_dloss == 0.0
    assert by_fmt["BF16"].predicted_dloss == 0.0
    # A positive weight-side price is a legitimate (biased) surrogate the DP
    # can trade off — only the exact zero is unbeatable.
    assert by_fmt["Q4_K"].predicted_dloss > 0.0

    # Visible: same counted/logged mechanism as the legality masks.
    assert [(r["qname"], r["format"], r["reason"]) for r in records] == [
        ("e", _ACTQUANT_FMT, ACTIVATION_COST_UNMEASURED_REASON)]
    summary = summarize_applicability_masks(records)
    assert summary["summary"][_ACTQUANT_FMT] == {
        ACTIVATION_COST_UNMEASURED_REASON: 1}
    assert "no measured output_mse" in records[0]["detail"]


def test_measured_output_mse_keeps_the_actquant_candidate():
    """The exclusion is about MISSING evidence, not about the format: the same
    weight-lossless W-and-A row with a real measured output_mse keeps its
    candidate and its measured A-side price."""
    stats = {"e": _packed_stats()}
    entry = {"weight_mse": 0.0, "output_mse": 6.3e-3, "rel_output_mse": 8.7e-3,
             "output_mse_measured": True}
    costs = {"e": {_ACTQUANT_FMT: entry}}
    cands = build_candidates(stats, costs, [fr.get_format(_ACTQUANT_FMT)])
    got = {c.fmt: c.predicted_dloss for c in cands["e"]}
    expected = 0.5 * stats["e"]["h_trace"] * entry["output_mse"]
    assert abs(got[_ACTQUANT_FMT] - expected) < 1e-12


def test_zero_sensitivity_rows_keep_every_format():
    """``h_trace == 0`` prices EVERY format at 0.0, passthrough ones included:
    that is a measured statement that no perturbation of this Linear's output
    moves the loss — the same Fisher factor multiplies the W and A sides — so
    the row must stay free to take the cheapest format. A zero-token expert at
    thin calibration is exactly this row; forcing it to BF16 would be a large
    bpp regression justified by nothing."""
    stats = {"e": _packed_stats(h_trace=0.0)}
    costs = {"e": {_ACTQUANT_FMT: _unmeasured_output_entry(0.0),
                   "Q4_K": _unmeasured_output_entry(2.0e-6),
                   "BF16": {"weight_mse": 0.0}}}
    specs = [fr.get_format(f) for f in (_ACTQUANT_FMT, "Q4_K", "BF16")]
    records: list[dict] = []
    cands = build_candidates(stats, costs, specs, mask_records=records)
    assert {c.fmt for c in cands["e"]} == {_ACTQUANT_FMT, "Q4_K", "BF16"}
    assert records == []


def test_starving_a_linear_of_all_candidates_is_a_hard_error():
    """The exclusion must never silently drop a Linear: a name absent from
    build_candidates' output never reaches the DP, so its bytes vanish from the
    bpp/footprint accounting and from serving-unit membership while export
    still emits the tensor. When every legal format for a row would have been
    priced at the unmeasured zero, the allocator says so instead."""
    import pytest

    stats = {"e": _packed_stats()}
    # Menu of activation-quantizing formats only, all weight-lossless.
    costs = {"e": {_ACTQUANT_FMT: _unmeasured_output_entry(0.0),
                   "Q4_K": _unmeasured_output_entry(0.0)}}
    specs = [fr.get_format(_ACTQUANT_FMT), fr.get_format("Q4_K")]
    with pytest.raises(AssertionError) as exc:
        build_candidates(stats, costs, specs)
    msg = str(exc.value)
    assert "e" in msg and _ACTQUANT_FMT in msg
    assert "PRISMAQUANT_EXPERT_COST_SAMPLE" in msg
    # Adding one identity-activation rung is a legal resolution.
    costs["e"][_PASSTHROUGH_FMT] = _unmeasured_output_entry(0.0)
    cands = build_candidates(
        stats, costs, specs + [fr.get_format(_PASSTHROUGH_FMT)])
    assert {c.fmt for c in cands["e"]} == {_PASSTHROUGH_FMT}


def test_unmeasured_zero_no_longer_wins_the_dp_at_every_budget():
    """End to end: with the pathology in place the weight-lossless W-and-A
    format is chosen at every target (zero cost, fewest bits). Excluded, the
    DP has to buy quality with bits again."""
    from prismaquant.allocator_solver import solve_with_promotion

    menu = ["MXFP4", "Q4_K", "MXFP8A16"]
    stats, costs = {}, {}
    for i in range(4):
        name = f"model.layers.{i}.mlp.experts.down_proj"
        stats[name] = _packed_stats()
        costs[name] = {
            "MXFP4": _unmeasured_output_entry(0.0),      # lossless re-encode
            "Q4_K": _unmeasured_output_entry(2.0e-6),
            "MXFP8A16": _unmeasured_output_entry(0.0),   # passthrough acts
        }
    specs = [fr.get_format(f) for f in menu]
    cands = build_candidates(stats, costs, specs)
    assert all("MXFP4" not in {c.fmt for c in v} for v in cands.values())

    spec_map = {f: fr.get_format(f) for f in menu}
    rank = {f: i for i, f in enumerate(
        sorted(menu, key=lambda f: fr.get_format(f).effective_bits))}
    assign, achieved = solve_with_promotion(
        stats, cands, 4.6, spec_map, rank, bit_precision=0.001)
    assert assign is not None
    assert set(assign.values()) == {"Q4_K"}, assign
    assert achieved <= 4.6 + 0.01


def test_packed_group_pricing_splits_passthrough_and_actquant():
    """Group level: members bit-exact at a passthrough-activation format
    sum to a zero group cost; the same members at a weight-lossless W·A·
    format carry the summed A-side cost."""
    from prismaquant.allocator_candidates import (
        _PACKED_GROUP_MARKER,
        aggregate_packed_serving_groups,
    )

    class _P:
        def packed_expert_format_group(self, name):
            return "g" if ".experts." in name else None

    n_members = 4
    stats, costs = {}, {}
    specs = [fr.get_format(_PASSTHROUGH_FMT), fr.get_format(_ACTQUANT_FMT)]
    for e in range(n_members):
        name = f"model.layers.0.mlp.experts.{e}.gate_proj"
        stats[name] = {"h_trace": 0.5, "n_params": 65536,
                       "in_features": 256, "out_features": 256}
        costs[name] = {_PASSTHROUGH_FMT: _bit_exact_entry(),
                       _ACTQUANT_FMT: _bit_exact_entry()}
    cands = build_candidates(stats, costs, specs)
    _stats2, _costs2, cands2 = aggregate_packed_serving_groups(
        stats, costs, specs, cands, _P())
    super_name = next(n for n in cands2 if _PACKED_GROUP_MARKER in n)
    by_fmt = {c.fmt: c for c in cands2[super_name]}
    assert by_fmt[_PASSTHROUGH_FMT].predicted_dloss == 0.0
    per_member = 0.5 * 0.5 * _bit_exact_entry()["output_mse"]
    assert abs(
        by_fmt[_ACTQUANT_FMT].predicted_dloss - n_members * per_member
    ) < 1e-12
