"""Serving context is explicit immutable identity, not a runtime default."""
from dataclasses import FrozenInstanceError

import pytest

from prismaquant import lane_eligibility as lane


def _context(**changed):
    fields = {
        "platform": "sm_121",
        "structure": "dense",
        "residency": "resident",
        "runtime_image": "example/runtime@sha256:" + "a" * 64,
        "execution_mode": "eager",
    }
    fields.update(changed)
    return lane.ServingContext(**fields)


def test_scope_identity_is_immutable_and_serializes_every_axis():
    context = _context()
    assert context.key() == tuple(context.as_dict().values())
    assert context.as_dict() == {
        "platform": "sm_121", "structure": "dense", "residency": "resident",
        "runtime_image": "example/runtime@sha256:" + "a" * 64,
        "execution_mode": "eager",
    }
    assert hash(context) == hash(_context())
    with pytest.raises(FrozenInstanceError):
        context.execution_mode = "compiled"


@pytest.mark.parametrize("field,value", [
    ("platform", "sm_120"), ("structure", "routed_moe"),
    ("residency", "streamed"),
    ("runtime_image", "example/runtime@sha256:" + "b" * 64),
    ("execution_mode", "compiled"),
])
def test_each_scope_axis_separates_cache_identity(field, value):
    assert _context(**{field: value}).key() != _context().key()


@pytest.mark.parametrize("field", [
    "platform", "structure", "residency", "runtime_image", "execution_mode",
])
def test_blank_scope_axes_are_refused_by_name(field):
    with pytest.raises(lane.LaneEligibilityError, match=field):
        _context(**{field: ""})


@pytest.mark.parametrize("field,value", [
    ("structure", "guessed_moe"), ("residency", "automatic"),
    ("runtime_image", "example/runtime:latest"),
    ("runtime_image", "sha256:" + "a" * 64),
    ("execution_mode", "automatic"),
])
def test_unknown_or_mutable_scope_is_refused(field, value):
    with pytest.raises(lane.LaneEligibilityError, match=field):
        _context(**{field: value})
