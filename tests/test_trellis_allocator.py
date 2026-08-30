from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
from itertools import permutations
import json
import math
import random
import sys

import pytest

from prismaquant.allocator_solver import (
    Candidate,
    selected_rung_dual_intervals,
    solve_allocation,
)
from prismaquant.format_registry import list_formats, list_producer_formats
from prismaquant.trellis_allocator import (
    TrellisLambdaChoice,
    adaptive_trellis_rate_surface,
    build_trellis_allocator_candidate,
    trellis_lambda_choices,
    trellis_local_repair_solver_menu,
    trellis_pareto_frontier,
    trellis_rate_distortion_hull,
    trellis_solver_candidate_menu,
)
from prismaquant.serving_profiles import ResolvedServingLane, serving_lane_route
from prismaquant.trellis_formats import (
    E2M1_FAMILY,
    LAYOUT_FIXED_QUOTA,
    LAYOUT_TIGHT_OFFSETS,
    TrellisFormatError,
    native_code_value,
)


def _alphabet(rate: int) -> list[int]:
    codes = list(range(1 << (rate + 1)))
    return sorted(
        codes,
        key=lambda code: (native_code_value(E2M1_FAMILY, code), code),
    )


def _fixed_candidate(
    rate: int,
    loss: float,
    *,
    stderr: float = 0.0,
    target_profile: str = "research",
    unit_name: str = "unit",
    shape: tuple[int, int] = (8, 256),
    layout: str = LAYOUT_FIXED_QUOTA,
    reverse_schedule: bool = False,
    variant_label: str | None = None,
):
    rows, columns = shape
    assert rows > 0 and columns % 256 == 0
    base, remainder = divmod(rate, 256)
    block = [base] * (256 - remainder) + [base + 1] * remainder
    if reverse_schedule:
        block.reverse()
    schedule = block * (columns // 256)
    shaped_rates = sorted({value for value in schedule if value < 4})
    return build_trellis_allocator_candidate(
        unit_name,
        shape,
        family=E2M1_FAMILY,
        body_rate_q256=rate,
        layout=layout,
        schedule=schedule,
        alphabets={value: _alphabet(value) for value in shaped_rates},
        predicted_dloss=loss,
        predicted_dloss_stderr=stderr,
        target_profile=target_profile,
        qname="model.layers.0.mlp.down_proj",
        variant_label=variant_label,
    )


def _ordinary_byte_candidate(
    memory_bytes: int,
    loss: float,
    *,
    stderr: float = 0.0,
    unit_name: str = "ordinary-byte-unit",
):
    """Build a real footprint whose explicit sidecar gives an exact byte x."""

    base_bytes = 116
    assert memory_bytes >= base_bytes
    return build_trellis_allocator_candidate(
        unit_name,
        (1, 8),
        family=E2M1_FAMILY,
        body_rate_q256=256,
        layout=LAYOUT_FIXED_QUOTA,
        schedule=[1] * 8,
        alphabets={1: _alphabet(1)},
        predicted_dloss=loss,
        predicted_dloss_stderr=stderr,
        target_profile="research",
        sidecar_header_bytes=memory_bytes - base_bytes,
    )


def _canonical_digest(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("pre_render_recipe_identity_sha256", None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_unattested_route_status_travels_with_research_candidate():
    record = _fixed_candidate(
        384, 0.09, target_profile="trellis_research_sm121"
    )
    lane = record.servability.serving_lane
    assert lane is not None
    assert lane.route_status == "unattested"
    assert lane.activation_contract == "e2m1_group16_ue4m3_static"
    assert record.as_dict()["servability"]["serving_lane"][
        "route_status"
    ] == "unattested"
    solver_candidate = record.to_solver_candidate()
    assert solver_candidate.serving_lane is not None
    assert solver_candidate.serving_lane.route_status == "unattested"

    e4_lane = serving_lane_route(
        "trellis_research_sm121", "TCQ_E4M3_R1024"
    )
    assert e4_lane is not None
    assert e4_lane.route_status == "unattested"
    assert e4_lane.activation_contract == "fp8_per_token_dynamic"


def _brute_force_choice(records, alpha: float):
    exact_alpha = Fraction.from_float(alpha)
    return min(
        records,
        key=lambda record: (
            Fraction.from_float(record.predicted_dloss_objective)
            + exact_alpha * record.memory_bytes,
            record.memory_bytes,
            record.identity_sha256,
        ),
    )


def test_candidate_uses_exact_payload_point_estimate_and_profile_gate():
    record = build_trellis_allocator_candidate(
        "unit",
        (8, 256),
        family=E2M1_FAMILY,
        body_rate_q256=512,
        layout=LAYOUT_TIGHT_OFFSETS,
        schedule=[2] * 256,
        alphabets={2: _alphabet(2)},
        predicted_dloss=0.010,
        predicted_dloss_stderr=0.001,
        target_profile="research",
    )

    assert record.servability.legal
    assert record.servability.profile_emulation_only
    assert record.predicted_dloss_objective == pytest.approx(0.010)
    assert record.predicted_dloss_stderr == pytest.approx(0.001)
    assert record.memory_bytes == 875
    assert record.bits_per_param == 3.41796875
    assert record.as_dict()["producer_eligible"] is False
    candidate = record.to_solver_candidate()
    assert candidate.memory_bytes == 875
    assert candidate.bits_per_param == 3.41796875
    assert candidate.predicted_dloss == pytest.approx(0.010)
    assert candidate.serialized_identity == record.footprint[
        "pre_render_recipe_identity_sha256"
    ]
    assert candidate.serialized_sidecar_identity is None
    assert record.as_dict()["rendered_wire_identity_sha256"] is None

    denied = _fixed_candidate(
        512,
        0.01,
        target_profile="nvfp4_cb",
    )
    assert not denied.servability.legal
    assert denied.servability.format_decision.reason in {
        "profile_mismatch",
        "exporter_cannot_emit",
    }
    with pytest.raises(TrellisFormatError, match="denied by target profile"):
        denied.to_solver_candidate()


def test_trellis_surface_creates_no_public_or_producer_format_ids():
    assert not any(spec.name.startswith("TCQ_") for spec in list_formats())
    assert not any(
        spec.name.startswith("TCQ_") for spec in list_producer_formats()
    )
    record = _fixed_candidate(512, 0.01)
    assert record.allocator_key.startswith("__TRELLIS_RESEARCH__:")
    assert all(spec.name != record.allocator_key for spec in list_formats())


def test_exact_trellis_candidates_feed_global_dp_and_dual_intervals():
    records = [
        _fixed_candidate(384, 0.090),
        _fixed_candidate(640, 0.035),
        _fixed_candidate(896, 0.020),
    ]
    menu = trellis_solver_candidate_menu(records)
    stats = {"unit": {"n_params": 8 * 256}}
    middle = records[1]
    solved = solve_allocation(
        stats,
        menu,
        target_bits=middle.bits_per_param,
        bit_precision=1.0e-6,
    )
    assert solved is not None
    assignment, selected = solved
    assert assignment == {"unit": middle.allocator_key}
    assert selected["unit"].memory_bytes == middle.memory_bytes

    intervals = selected_rung_dual_intervals(
        stats,
        menu,
        assignment,
        bit_precision=1.0e-6,
    )
    assert not intervals["unit"].is_empty
    assert intervals["unit"].lambda_lo >= 0.0

    # The bridge returns ordinary Candidate objects; native/CB ablation rungs
    # may be appended by the experiment without a Trellis family cutoff.
    menu["unit"].append(Candidate("NATIVE_ABLATION", 16.0, 4096, 0.0))
    assert {candidate.fmt for candidate in menu["unit"]} >= {
        middle.allocator_key,
        "NATIVE_ABLATION",
    }


def test_pareto_marginals_and_alpha_adaptive_surface_are_reproducible():
    records = [
        _fixed_candidate(384, 0.090, stderr=0.002),
        _fixed_candidate(640, 0.035, stderr=0.001),
        _fixed_candidate(896, 0.020, stderr=0.001),
    ]
    frontier = trellis_pareto_frontier(records, uncertainty_z=1.0)
    assert [candidate.body_rate_q256 for candidate in frontier.candidates] == [
        384, 640, 896,
    ]
    first = frontier.segments[0]
    assert first.delta_bytes == (
        records[1].memory_bytes - records[0].memory_bytes
    )
    assert (
        first.objective_marginal_loss_per_byte_rounded_binary64_diagnostic
        == pytest.approx((0.090 - 0.035) / first.delta_bytes)
    )
    assert first.uncertainty_low_loss_per_byte == pytest.approx(
        (0.090 - 0.035 - 0.003) / first.delta_bytes
    )
    assert first.uncertainty_high_loss_per_byte == pytest.approx(
        (0.090 - 0.035 + 0.003) / first.delta_bytes
    )

    proposal = adaptive_trellis_rate_surface(
        E2M1_FAMILY,
        records,
        alpha_loss_per_byte=(
            first.objective_marginal_loss_per_byte_exact
            .rounded_binary64_diagnostic
        ),
        max_new_points=1,
        uncertainty_z=1.0,
    )
    assert proposal.surface.anchor_q256 == (384, 640, 896)
    assert proposal.surface.proposed_q256 == (512,)
    assert tuple(proposal.surface.rates_q256) == (384, 512, 640, 896)
    assert proposal.ranked_brackets[0][
        "alpha_bracketed_by_uncertainty"
    ] is True
    assert set(proposal.ranked_brackets[0]["alpha_point_distance_exact"]) == {
        "numerator_decimal",
        "denominator_decimal",
        "rounded_binary64_diagnostic",
    }
    assert proposal.ranked_brackets[0]["selected_for_refinement"] is True
    repeated = adaptive_trellis_rate_surface(
        E2M1_FAMILY,
        records,
        alpha_loss_per_byte=(
            first.objective_marginal_loss_per_byte_exact
            .rounded_binary64_diagnostic
        ),
        max_new_points=1,
        uncertainty_z=1.0,
    )
    assert proposal.identity_sha256 == repeated.identity_sha256


def test_solver_menu_drops_denied_records_but_can_fail_closed_explicitly():
    legal = _fixed_candidate(384, 0.09)
    denied = _fixed_candidate(512, 0.04, target_profile="nvfp4_cb")
    with pytest.raises(TrellisFormatError, match="mixes target profiles"):
        trellis_solver_candidate_menu([legal, denied])

    # Filtering remains available across distinct allocator units, where the
    # per-unit shape/profile domain is unambiguous.
    denied_other_unit = replace(denied, unit_name="denied_unit")
    assert list(trellis_solver_candidate_menu([
        legal, denied_other_unit,
    ])) == ["unit"]
    assert len(trellis_solver_candidate_menu([
        legal, denied_other_unit,
    ])["unit"]) == 1
    with pytest.raises(TrellisFormatError, match="denied by target profile"):
        trellis_solver_candidate_menu(
            [legal, denied_other_unit],
            fail_on_denied=True,
        )


def test_per_tensor_rd_hull_has_log_lambda_lookup_and_uncertified_repair_menu():
    # R512 is Pareto but above the lower convex envelope: no scalar lambda
    # selects it, while integer-budget repair may still need it.
    records = [
        _fixed_candidate(384, 0.090),
        _fixed_candidate(512, 0.085),
        _fixed_candidate(640, 0.035),
        _fixed_candidate(896, 0.020),
    ]
    frontier = trellis_pareto_frontier(records)
    hull = trellis_rate_distortion_hull(records)
    assert [candidate.body_rate_q256 for candidate in frontier.candidates] == [
        384, 512, 640, 896,
    ]
    assert [candidate.body_rate_q256 for candidate in hull.candidates] == [
        384, 640, 896,
    ]
    assert hull.cheapest_legal_floor.body_rate_q256 == 384
    assert hull.as_dict()["cheapest_legal_floor"]["memory_bytes"] == (
        records[0].memory_bytes
    )
    benefits = hull.exact_breakpoints_loss_per_byte
    assert all(
        left.compare(right) > 0
        for left, right in zip(benefits, benefits[1:])
    )

    rounded_first = benefits[0].rounded_binary64_diagnostic
    at_breakpoint = hull.choice_at_lambda(rounded_first)
    assert at_breakpoint.candidate.identity_sha256 == _brute_force_choice(
        records,
        rounded_first,
    ).identity_sha256
    between_lambda = (
        benefits[0].rounded_binary64_diagnostic
        + benefits[1].rounded_binary64_diagnostic
    ) / 2.0
    between = hull.choice_at_lambda(between_lambda)
    assert between.candidate.identity_sha256 == _brute_force_choice(
        records,
        between_lambda,
    ).identity_sha256
    richest = hull.choice_at_lambda(0.0)
    assert richest.candidate.body_rate_q256 == 896
    comparison_bound = math.ceil(math.log2(len(hull.candidates)))
    assert hull.as_dict()["maximum_lambda_lookup_comparisons"] == (
        comparison_bound
    )
    assert hull.as_dict()["exact_breakpoints_precomputed"] is True
    cached_breakpoints = hull.exact_breakpoints_loss_per_byte
    assert max(
        at_breakpoint.comparisons,
        between.comparisons,
        richest.comparisons,
    ) <= comparison_bound
    assert hull.exact_breakpoints_loss_per_byte is cached_breakpoints
    assert trellis_lambda_choices(
        {"unit": hull},
        between.lambda_loss_per_byte,
    )["unit"].candidate.identity_sha256 == between.candidate.identity_sha256

    local = trellis_local_repair_solver_menu(
        frontier,
        between,
        neighbor_count=1,
    )
    assert [candidate.fmt for candidate in local["unit"]] == [
        records[1].allocator_key,
        records[2].allocator_key,
        records[3].allocator_key,
    ]
    complete = trellis_local_repair_solver_menu(
        frontier,
        between,
        complete_pareto=True,
    )
    assert len(complete["unit"]) == 4

    uncertain_records = [
        _fixed_candidate(384, 0.090, stderr=0.10),
        _fixed_candidate(640, 0.035, stderr=0.10),
        _fixed_candidate(896, 0.020, stderr=0.10),
    ]
    uncertain_frontier = trellis_pareto_frontier(uncertain_records)
    uncertain_hull = trellis_rate_distortion_hull(uncertain_records)
    uncertain_choice = uncertain_hull.choice_at_lambda(
        uncertain_hull.exact_breakpoints_loss_per_byte[
            0
        ].rounded_binary64_diagnostic
    )
    overlap_menu = trellis_local_repair_solver_menu(
        uncertain_frontier,
        uncertain_choice,
        neighbor_count=0,
        include_ci_overlap=True,
    )
    assert len(overlap_menu["unit"]) > 1


def test_ci_overlap_uses_centered_differences_and_rejects_nonfinite_derivations():
    records = [
        _ordinary_byte_candidate(271, 2.0),
        _ordinary_byte_candidate(276, 1.0),
        _ordinary_byte_candidate(279, 0.0),
    ]
    frontier = trellis_pareto_frontier(records, uncertainty_z=0.0)
    hull = trellis_rate_distortion_hull(records, uncertainty_z=0.0)

    # Absolute loss + lambda*bytes would overflow at all three points. The
    # centered delta is finite and therefore remains a valid overlap test.
    centered_choice = hull.choice_at_lambda(1.0e307)
    centered_menu = trellis_local_repair_solver_menu(
        frontier,
        centered_choice,
        neighbor_count=0,
        include_ci_overlap=True,
        ci_uncertainty_z=0.0,
    )
    assert len(centered_menu[frontier.unit_name]) == 1

    overflowing_price = hull.choice_at_lambda(sys.float_info.max)
    with pytest.raises(
        TrellisFormatError,
        match=r"lambda \* delta_bytes",
    ):
        trellis_local_repair_solver_menu(
            frontier,
            overflowing_price,
            neighbor_count=0,
            include_ci_overlap=True,
            ci_uncertainty_z=0.0,
        )

    large_stderr_records = [
        _ordinary_byte_candidate(271, 2.0, stderr=8.0e307),
        _ordinary_byte_candidate(276, 1.0, stderr=8.0e307),
    ]
    large_frontier = trellis_pareto_frontier(
        large_stderr_records,
        uncertainty_z=0.0,
    )
    large_hull = trellis_rate_distortion_hull(
        large_stderr_records,
        uncertainty_z=0.0,
    )
    with pytest.raises(TrellisFormatError, match=r"z \* chosen stderr"):
        trellis_local_repair_solver_menu(
            large_frontier,
            large_hull.choice_at_lambda(0.0),
            neighbor_count=0,
            include_ci_overlap=True,
            ci_uncertainty_z=3.0,
        )

    base_frontier = trellis_pareto_frontier(records, uncertainty_z=0.0)
    radius_candidates = tuple(
        replace(candidate, predicted_dloss_stderr=9.0e307)
        for candidate in base_frontier.candidates
    )
    radius_frontier = replace(
        base_frontier,
        candidates=radius_candidates,
    )
    radius_choice = TrellisLambdaChoice(
        candidate=radius_candidates[0],
        hull_index=0,
        lambda_loss_per_byte=0.0,
        comparisons=0,
        cheaper_breakpoint_loss_per_byte_exact=None,
        dearer_breakpoint_loss_per_byte_exact=None,
    )
    with pytest.raises(TrellisFormatError, match="uncertainty-radius sum"):
        trellis_local_repair_solver_menu(
            radius_frontier,
            radius_choice,
            neighbor_count=0,
            include_ci_overlap=True,
            ci_uncertainty_z=1.0,
        )


def test_candidate_and_adaptive_metadata_are_recursively_immutable_copies():
    record = _fixed_candidate(384, 0.09)
    original_identity = record.identity_sha256
    with pytest.raises(TypeError):
        record.footprint["total_bytes"] = 1
    with pytest.raises(TypeError):
        record.footprint["alphabet_bytes_by_rate"]["1"] = 1
    with pytest.raises(TypeError):
        record.footprint["shape"][0] = 1

    serialized = record.as_dict()
    serialized["footprint"]["shape"][0] = 999
    serialized["footprint"]["alphabet_bytes_by_rate"]["1"] = 999
    serialized["servability"]["format"]["detail"] = "mutated copy"
    assert record.shape == (8, 256)
    assert record.identity_sha256 == original_identity

    proposal = adaptive_trellis_rate_surface(
        E2M1_FAMILY,
        [_fixed_candidate(384, 0.09), _fixed_candidate(640, 0.035)],
    )
    proposal_identity = proposal.identity_sha256
    with pytest.raises(TypeError):
        proposal.ranked_brackets[0]["selected_for_refinement"] = False
    with pytest.raises(TypeError):
        proposal.ranked_brackets[0]["q256_interval"][0] = 0
    proposal_copy = proposal.as_dict()
    proposal_copy["ranked_brackets"][0]["q256_interval"][0] = 0
    proposal_copy["surface"]["anchor_q256"][0] = 0
    assert proposal.identity_sha256 == proposal_identity


def test_candidate_revalidates_footprint_digest_schema_and_byte_arithmetic():
    record = _fixed_candidate(384, 0.09)
    stale = record.as_dict()["footprint"]
    stale["total_bytes"] += 1
    with pytest.raises(TrellisFormatError, match="identity does not match"):
        replace(record, footprint=stale)

    self_resigned = record.as_dict()["footprint"]
    self_resigned["total_bytes"] += 1
    self_resigned["pre_render_recipe_identity_sha256"] = _canonical_digest(
        self_resigned
    )
    with pytest.raises(TrellisFormatError, match="total-byte arithmetic"):
        replace(record, footprint=self_resigned)

    wrong_schema = record.as_dict()["footprint"]
    wrong_schema["schema"] = "attacker.trellis.v9"
    wrong_schema["pre_render_recipe_identity_sha256"] = _canonical_digest(
        wrong_schema
    )
    with pytest.raises(TrellisFormatError, match="unsupported payload schema"):
        replace(record, footprint=wrong_schema)


def test_candidate_copies_all_nested_resolved_lane_sequences():
    record = _fixed_candidate(384, 0.09)
    mutable_rungs = [40, 44, 48]
    mutable_range = [40, 48]
    mutable_flags = ["FLAG_A", "FLAG_B"]
    lane = ResolvedServingLane(
        lane_id="test-lane",
        format=record.footprint["format"],
        activation_contract="test",
        fallback_route="test-fallback",
        fused_mid_m_backed=True,
        fused_mid_m_rungs=mutable_rungs,
        fused_mid_m_range=mutable_range,
        runtime_version="test-runtime",
        rungs_source="test-source",
        requires_serve_flags=mutable_flags,
    )
    candidate = replace(
        record,
        servability=replace(record.servability, serving_lane=lane),
    )
    identity = candidate.identity_sha256
    mutable_rungs.append(52)
    mutable_range[0] = 0
    mutable_flags.append("FLAG_C")

    copied_lane = candidate.servability.serving_lane
    assert copied_lane is not None
    assert copied_lane.fused_mid_m_rungs == (40, 44, 48)
    assert copied_lane.fused_mid_m_range == (40, 48)
    assert copied_lane.requires_serve_flags == ("FLAG_A", "FLAG_B")
    assert candidate.identity_sha256 == identity

    serialized = candidate.as_dict()
    serialized["servability"]["serving_lane"][
        "fused_mid_m_rungs"
    ].append(99)
    assert candidate.identity_sha256 == identity


def test_recipe_addresses_are_collision_free_and_duplicate_measurements_refuse():
    first = _fixed_candidate(384, 0.09)
    reordered = _fixed_candidate(
        384,
        0.09,
        reverse_schedule=True,
        variant_label="alternate-schedule",
    )
    assert first.pre_render_recipe_identity_sha256 != (
        reordered.pre_render_recipe_identity_sha256
    )
    assert first.allocator_key != reordered.allocator_key
    assert ":recipe=" in first.allocator_key
    assert first.allocator_key.endswith(first.pre_render_recipe_identity_sha256)
    assert reordered.allocator_key.endswith(
        reordered.pre_render_recipe_identity_sha256
    )
    assert len(trellis_solver_candidate_menu([first, reordered])["unit"]) == 2

    relabeled_duplicate = replace(first, variant_label="repeat")
    assert relabeled_duplicate.identity_sha256 != first.identity_sha256
    assert relabeled_duplicate.allocator_key == first.allocator_key
    for function in (trellis_solver_candidate_menu, trellis_pareto_frontier):
        with pytest.raises(
            TrellisFormatError,
            match="duplicate Trellis pre-render recipe/address",
        ):
            function([first, relabeled_duplicate])


def test_candidate_curve_order_and_identities_are_permutation_invariant():
    records = (
        _fixed_candidate(384, 0.090),
        _fixed_candidate(
            384, 0.090, reverse_schedule=True,
            variant_label="alternate-schedule",
        ),
        _fixed_candidate(640, 0.035),
        _fixed_candidate(896, 0.020),
    )
    expected: tuple[object, ...] | None = None
    for ordering in permutations(records):
        menu = trellis_solver_candidate_menu(ordering)["unit"]
        frontier = trellis_pareto_frontier(ordering)
        hull = trellis_rate_distortion_hull(ordering)
        proposal = adaptive_trellis_rate_surface(
            E2M1_FAMILY,
            ordering,
            max_new_points=3,
        )
        observed = (
            tuple(candidate.fmt for candidate in menu),
            frontier.identity_sha256,
            hull.identity_sha256,
            proposal.identity_sha256,
        )
        if expected is None:
            expected = observed
        assert observed == expected


def test_same_unit_shape_profile_and_adaptive_layout_domains_are_closed():
    reference = _fixed_candidate(384, 0.09)
    same_nparams_different_shape = _fixed_candidate(
        640,
        0.04,
        shape=(4, 512),
    )
    with pytest.raises(TrellisFormatError, match="mixes tensor shapes/n_params"):
        trellis_solver_candidate_menu([
            reference, same_nparams_different_shape,
        ])

    different_nparams = _fixed_candidate(640, 0.04, shape=(4, 256))
    with pytest.raises(TrellisFormatError, match="mixes tensor shapes/n_params"):
        trellis_pareto_frontier([reference, different_nparams])

    other_profile = _fixed_candidate(
        640, 0.04, target_profile="nvfp4_cb",
    )
    with pytest.raises(TrellisFormatError, match="mixes target profiles"):
        trellis_solver_candidate_menu([reference, other_profile])

    tight = _fixed_candidate(
        640, 0.04, layout=LAYOUT_TIGHT_OFFSETS,
    )
    # Exact-byte global menus may compare layouts as ordinary candidates.
    assert len(trellis_solver_candidate_menu([reference, tight])["unit"]) == 2
    with pytest.raises(TrellisFormatError, match="must use one Trellis layout"):
        adaptive_trellis_rate_surface(E2M1_FAMILY, [reference, tight])


def test_adaptive_refinement_subdivides_observed_off_hull_midpoints():
    # R512 is objective-Pareto but lies above the 384->640 lower hull segment.
    records = [
        _fixed_candidate(384, 0.090),
        _fixed_candidate(512, 0.085),
        _fixed_candidate(640, 0.035),
    ]
    first = adaptive_trellis_rate_surface(
        E2M1_FAMILY,
        records,
        max_new_points=1,
    )
    assert first.surface.proposed_q256 == (448,)
    assert first.ranked_brackets[0]["parent_hull_q256_interval"] == (
        384, 640,
    )
    assert first.ranked_brackets[0][
        "parent_anchor_index_interval_inclusive"
    ] == (0, 2)

    # Once the selected point is measured, the largest remaining observed
    # sub-bracket advances deterministically instead of retrying R512.
    second = adaptive_trellis_rate_surface(
        E2M1_FAMILY,
        [*records, _fixed_candidate(448, 0.088)],
        max_new_points=1,
    )
    assert second.surface.proposed_q256 == (576,)

    capped = adaptive_trellis_rate_surface(
        E2M1_FAMILY,
        records,
        max_new_points=5,
    )
    assert len(capped.surface.proposed_q256) == 5
    assert len(set(capped.surface.proposed_q256)) == 5
    # Initial measured subdivisions plus at most two queue rows per emitted
    # point: a deterministic O(P + H + max_new_points) space bound.
    hull = trellis_rate_distortion_hull(records)
    assert len(capped.ranked_brackets) <= (
        len(records) + len(hull.segments) + 2 * 5
    )

    exhausted = adaptive_trellis_rate_surface(
        E2M1_FAMILY,
        [_fixed_candidate(384, 0.09), _fixed_candidate(385, 0.08)],
        max_new_points=8,
    )
    assert exhausted.surface.proposed_q256 == ()
    assert exhausted.ranked_brackets == ()


def test_adaptive_bracket_metadata_is_linear_not_parent_interior_quadratic():
    raw_records = [
        _fixed_candidate(rate, 1.0)
        for rate in range(384, 641, 2)
    ]
    # A concave, strictly decreasing measured curve leaves every rate observed
    # while the compressed lower hull has only its two endpoints.
    records = [
        replace(
            record,
            predicted_dloss_mean=math.sqrt(640 - record.body_rate_q256),
        )
        for record in raw_records
    ]
    hull = trellis_rate_distortion_hull(records)
    assert len(records) == 129
    assert len(hull.candidates) == 2

    proposal = adaptive_trellis_rate_surface(
        E2M1_FAMILY,
        records,
        max_new_points=0,
    )
    assert len(proposal.ranked_brackets) == 128
    assert all(
        "observed_interior_q256" not in row
        for row in proposal.ranked_brackets
    )
    assert {
        row["parent_anchor_index_interval_inclusive"]
        for row in proposal.ranked_brackets
    } == {(0, 128)}
    assert sum(
        len(row["parent_anchor_index_interval_inclusive"])
        for row in proposal.ranked_brackets
    ) == 256


def test_exact_marginal_handles_subnormal_and_uncertainty_refuses_overflow():
    subnormal = [
        _ordinary_byte_candidate(271, 5e-324),
        _ordinary_byte_candidate(279, 0.0),
    ]
    hull = trellis_rate_distortion_hull(subnormal)
    breakpoint = hull.exact_breakpoints_loss_per_byte[0]
    assert breakpoint.rounded_binary64_diagnostic == 0.0
    assert hull.choice_at_lambda(0.0).candidate.memory_bytes == 279
    assert hull.choice_at_lambda(5e-324).candidate.memory_bytes == 271

    overflowing_uncertainty = [
        _fixed_candidate(384, 0.09, stderr=1e308),
        _fixed_candidate(640, 0.03, stderr=1e308),
    ]
    with pytest.raises(TrellisFormatError, match="uncertainty stderr sum"):
        trellis_pareto_frontier(overflowing_uncertainty)


def test_exact_hull_keeps_ordinary_byte_point_hidden_by_equal_rounded_slopes():
    records = [
        _ordinary_byte_candidate(
            271,
            float.fromhex("0x1.7a7576c8960d7p-84"),
        ),
        _ordinary_byte_candidate(
            276,
            float.fromhex("0x1.1bd81916708a1p-85"),
        ),
        _ordinary_byte_candidate(279, 0.0),
    ]
    hull = trellis_rate_distortion_hull(records)
    assert [candidate.memory_bytes for candidate in hull.candidates] == [
        271, 276, 279,
    ]
    first, second = hull.exact_breakpoints_loss_per_byte
    assert first.compare(second) > 0
    assert first.rounded_binary64_diagnostic == (
        second.rounded_binary64_diagnostic
    )
    shared_rounded = first.rounded_binary64_diagnostic
    # The binary64 diagnostic lies strictly between the exact thresholds. A
    # rounded-slope hull drops the middle point and gets this argmin wrong.
    assert hull.choice_at_lambda(shared_rounded).candidate.memory_bytes == 276
    assert _brute_force_choice(records, shared_rounded).memory_bytes == 276

    exact_tie_records = [
        _ordinary_byte_candidate(
            271, 2.0, unit_name="exact-binary64-threshold"
        ),
        _ordinary_byte_candidate(
            272, 1.0, unit_name="exact-binary64-threshold"
        ),
    ]
    exact_tie_hull = trellis_rate_distortion_hull(exact_tie_records)
    assert exact_tie_hull.choice_at_lambda(1.0).candidate.memory_bytes == 271
    assert exact_tie_hull.choice_at_lambda(
        math.nextafter(1.0, 0.0)
    ).candidate.memory_bytes == 272
    assert exact_tie_hull.choice_at_lambda(
        math.nextafter(1.0, math.inf)
    ).candidate.memory_bytes == 271


def test_randomized_exact_rational_hulls_and_lambda_choices_match_bruteforce():
    rng = random.Random(0x7E11_15)
    for curve_index in range(24):
        point_count = rng.randint(4, 11)
        memory = 180
        memories: list[int] = []
        for _ in range(point_count):
            memory += rng.randint(1, 11)
            memories.append(memory)
        numerators = sorted(
            rng.sample(range(1, 1 << 22), point_count),
            reverse=True,
        )
        exponent = rng.randint(20, 90)
        losses = [math.ldexp(value, -exponent) for value in numerators]
        records = [
            _ordinary_byte_candidate(
                memory_bytes,
                loss,
                unit_name=f"random-exact-{curve_index}",
            )
            for memory_bytes, loss in zip(memories, losses)
        ]

        expected = []
        for record in records:
            while len(expected) >= 2:
                earlier, previous = expected[-2:]
                previous_slope = (
                    Fraction.from_float(
                        earlier.predicted_dloss_objective
                    )
                    - Fraction.from_float(
                        previous.predicted_dloss_objective
                    )
                ) / (previous.memory_bytes - earlier.memory_bytes)
                candidate_slope = (
                    Fraction.from_float(
                        previous.predicted_dloss_objective
                    )
                    - Fraction.from_float(record.predicted_dloss_objective)
                ) / (record.memory_bytes - previous.memory_bytes)
                if previous_slope > candidate_slope:
                    break
                expected.pop()
            expected.append(record)

        hull = trellis_rate_distortion_hull(records)
        assert [record.identity_sha256 for record in hull.candidates] == [
            record.identity_sha256 for record in expected
        ]
        for cheaper, dearer, exact in zip(
            hull.candidates,
            hull.candidates[1:],
            hull.exact_breakpoints_loss_per_byte,
        ):
            expected_ratio = (
                Fraction.from_float(cheaper.predicted_dloss_objective)
                - Fraction.from_float(dearer.predicted_dloss_objective)
            ) / (dearer.memory_bytes - cheaper.memory_bytes)
            assert Fraction(exact.numerator, exact.denominator) == expected_ratio

        lambdas = [0.0, sys.float_info.max]
        lambdas.extend(
            math.ldexp(rng.randrange(0, 1 << 22), -rng.randint(10, 100))
            for _ in range(20)
        )
        lambdas.extend(
            exact.rounded_binary64_diagnostic
            for exact in hull.exact_breakpoints_loss_per_byte
        )
        for alpha in lambdas:
            observed = hull.choice_at_lambda(alpha).candidate
            expected_choice = _brute_force_choice(records, alpha)
            assert observed.identity_sha256 == expected_choice.identity_sha256


def test_huge_finite_exact_lambda_choices_match_fraction_bruteforce():
    maximum = sys.float_info.max
    records = [
        _ordinary_byte_candidate(271, maximum),
        _ordinary_byte_candidate(276, maximum / 2.0),
        _ordinary_byte_candidate(279, 0.0),
    ]
    hull = trellis_rate_distortion_hull(records)
    lambdas = [
        0.0,
        5e-324,
        maximum / 16.0,
        maximum / 4.0,
        maximum,
    ]
    for alpha in lambdas:
        assert hull.choice_at_lambda(alpha).candidate.identity_sha256 == (
            _brute_force_choice(records, alpha).identity_sha256
        )


def test_hull_copies_candidate_segment_lists_and_keeps_cached_breakpoints():
    hull = trellis_rate_distortion_hull([
        _fixed_candidate(384, 0.09),
        _fixed_candidate(640, 0.035),
        _fixed_candidate(896, 0.02),
    ])
    candidates = list(hull.candidates)
    segments = list(hull.segments)
    copied = replace(hull, candidates=candidates, segments=segments)
    identity = copied.identity_sha256
    breakpoints = copied.exact_breakpoints_loss_per_byte
    candidates.clear()
    segments.clear()
    assert copied.candidates == hull.candidates
    assert copied.segments == hull.segments
    assert copied.exact_breakpoints_loss_per_byte is breakpoints
    assert copied.identity_sha256 == identity
