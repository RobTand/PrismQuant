import pytest
import torch

from prismaquant.render_score import (
    RenderMechanismSpec,
    gate_render_candidate,
    persisted_cell_score_fields,
    register_render_mechanism,
    resolve_render_mechanism_order,
    score_render_error,
)


def test_persisted_cell_scores_are_optional_and_validate_both_sources():
    assert persisted_cell_score_fields(None) == {}
    assert persisted_cell_score_fields({}) == {}

    fields = persisted_cell_score_fields({
        "activation_output_mse": 0.25,
        "activation_output_mse_by_codebook_source": {
            "learned": 0.125,
            "lattice": 0.25,
        },
    })
    assert fields == {
        "activation_output_mse": 0.25,
        "activation_output_mse_by_codebook_source": {
            "lattice": 0.25,
            "learned": 0.125,
        },
    }

    with pytest.raises(ValueError, match="unknown source"):
        persisted_cell_score_fields({
            "activation_output_mse_by_codebook_source": {"other": 1.0},
        })
    with pytest.raises(ValueError, match="finite nonnegative"):
        persisted_cell_score_fields({"activation_output_mse": float("nan")})


def test_score_render_error_prefers_lower_output_error():
    weight = torch.eye(2)
    acts = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    good = weight + torch.tensor([[0.01, 0.0], [0.0, 0.01]])
    bad = weight + torch.tensor([[0.5, 0.0], [0.0, 0.5]])

    assert score_render_error(weight, good, acts) < score_render_error(
        weight, bad, acts
    )


def test_score_render_error_accepts_fisher_row_weights():
    weight = torch.eye(2)
    acts = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    candidate_a = weight + torch.tensor([[0.4, 0.0], [0.0, 0.01]])
    candidate_b = weight + torch.tensor([[0.01, 0.0], [0.0, 0.4]])

    unweighted_a = score_render_error(weight, candidate_a, acts)
    unweighted_b = score_render_error(weight, candidate_b, acts)
    assert abs(unweighted_a - unweighted_b) < 1e-5

    row_weights = torch.tensor([32.0, 1.0])
    assert score_render_error(
        weight, candidate_b, acts, row_weights=row_weights
    ) < score_render_error(
        weight, candidate_a, acts, row_weights=row_weights
    )


def test_gate_render_candidate_rejects_regression_and_min_gain():
    accepted = gate_render_candidate(
        baseline_score=10.0,
        candidate_score=8.0,
        metric="fisher_output_mse",
        min_relative_gain=0.1,
    )
    assert accepted.accepted
    assert accepted.relative_gain == 0.2

    tiny = gate_render_candidate(
        baseline_score=10.0,
        candidate_score=9.9,
        metric="fisher_output_mse",
        min_relative_gain=0.02,
    )
    assert not tiny.accepted
    assert tiny.reason == "below_min_gain"

    regression = gate_render_candidate(
        baseline_score=10.0,
        candidate_score=11.0,
        metric="fisher_output_mse",
    )
    assert not regression.accepted
    assert regression.reason == "regressed_or_tied"


def test_builtin_mechanism_order_uses_operation_semantics():
    plan = resolve_render_mechanism_order([
        "scale_sweep",
        "gptq",
        "fisher_gptq",
        "four_over_six",
    ])

    assert plan.errors == ()
    assert plan.names() == (
        "four_over_six",
        "fisher_gptq",
        "gptq",
        "scale_sweep",
    )


def test_plugin_mechanism_can_declare_phase_and_dependency():
    register_render_mechanism(RenderMechanismSpec(
        name="unit_test_post_scale_polish",
        operation="post_scale_polish",
        scope="linear",
        phase=70,
        gate_metric="fisher_output_mse",
        after=("scale_sweep",),
    ))

    plan = resolve_render_mechanism_order([
        "unit_test_post_scale_polish",
        "gptq",
        "scale_sweep",
    ])

    assert plan.errors == ()
    assert plan.names() == ("gptq", "scale_sweep", "unit_test_post_scale_polish")
