"""PQ #261: the streamed harness must retain the unit's own activation source.

``tessera_render._encoder_kwargs_for_plane`` builds an empty source and an
identity-probe source during ``encode_tessera_unit`` -- after the renderer
built the unit's Hessian source -- so a "last source wins" recorder handed the
probe to ``retain_production_wire`` and Tessera refused ("cached unit has no
exact Hessian key").
"""
import importlib.util
from pathlib import Path


def _harness():
    path = Path(__file__).resolve().parents[1] / "experiments" / "pq237_joint_aura_streamed.py"
    spec = importlib.util.spec_from_file_location("pq237_joint_aura_streamed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_activation_source(hessians, identity, **overrides):
    return {"hessians": dict(hessians), "identity": identity}


def test_recorder_keeps_the_unit_source_over_later_unit_less_sources():
    recorder = _harness().unit_activation_source_recorder
    captured = {"unit": "model.layers.0.mlp.down_proj"}
    record = recorder(captured, _fake_activation_source)
    unit = record({"model.layers.0.mlp.down_proj": "H"}, "id")
    record({}, "id")                      # _encoder_kwargs_for_plane: width query
    record({"probe": "eye"}, "id")        # _encoder_kwargs_for_plane: identity probe
    assert captured["value"] is unit
    assert set(captured["value"]["hessians"]) == {"model.layers.0.mlp.down_proj"}


def test_recorder_records_nothing_without_the_unit_hessian():
    recorder = _harness().unit_activation_source_recorder
    captured = {"unit": "model.layers.0.mlp.down_proj"}
    record = recorder(captured, _fake_activation_source)
    record({}, "id")
    record({"probe": "eye"}, "id")
    record({"model.layers.1.mlp.down_proj": "H"}, "id")
    assert "value" not in captured


def test_recorder_returns_the_real_source_every_call():
    recorder = _harness().unit_activation_source_recorder
    calls = []

    def source(hessians, identity, **overrides):
        calls.append((dict(hessians), identity, overrides))
        return len(calls)

    record = recorder({"unit": "u"}, source)
    assert record({"u": 1}, "id", ldlq_block=4) == 1
    assert record({}, "id") == 2
    assert calls[0] == ({"u": 1}, "id", {"ldlq_block": 4})
