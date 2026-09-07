"""Invalid Fisher frames cannot publish biased stack prices as a census."""
import math

import pytest

from prismaquant.tessera_campaign import (
    StackExpertSample, StackSampleError, _validate_stack_sample, draw_stack_sample,
)


def sample(**overrides):
    values = dict(packed_qname="experts.down_proj", packed_experts_module="experts",
                  packed_param="down_proj", num_experts=3, stack_h_trace=3.0,
                  h_trace_per_expert=(1.0, 1.0, 1.0), sampled_experts=(0, 1),
                  inclusion_prob={0: 2/3, 1: 2/3},
                  members={0: ("experts.0.w2",), 1: ("experts.1.w2",)}, seed=0)
    values.update(overrides)
    return StackExpertSample(**values)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_sampler_refuses_nonfinite_fisher_before_census(bad):
    with pytest.raises(RuntimeError, match="finite"):
        draw_stack_sample({"a": 1.0, "b": bad}, 2, seed=0, stack="s")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_stack_refuses_invalid_unsampled_fisher(bad):
    with pytest.raises(StackSampleError, match="finite|negative"):
        _validate_stack_sample(sample(h_trace_per_expert=(2.0, 2.0, bad)))


def test_stack_refuses_infinite_total():
    with pytest.raises(StackSampleError, match="finite"):
        _validate_stack_sample(sample(stack_h_trace=math.inf))


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.1, math.nan, math.inf])
def test_supplied_unsampled_probability_is_validated(bad):
    with pytest.raises(StackSampleError, match="cannot be drawn|outside"):
        _validate_stack_sample(sample(inclusion_prob={0: 2/3, 1: 2/3, 2: bad}))


def test_supplied_certainty_expert_cannot_be_absent():
    with pytest.raises(StackSampleError, match="certainty"):
        _validate_stack_sample(sample(inclusion_prob={0: 0.5, 1: 0.5, 2: 1.0}))


def test_zero_weight_unsampled_expert_is_valid():
    _validate_stack_sample(sample(stack_h_trace=2.0,
                                 h_trace_per_expert=(1.0, 1.0, 0.0),
                                 inclusion_prob={0: 1.0, 1: 1.0, 2: 0.0}))


def test_sampled_only_probability_contract_stays_supported():
    _validate_stack_sample(sample())
