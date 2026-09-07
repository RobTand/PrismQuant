"""Batch admission preserves per-unit inputs, prices, wire bytes and retries."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_render as render


FMT = "TESSERA_E4M3_K1_R1024"


def test_pending_batches_keep_each_anchor_once_and_split_incompatible_units():
    weights = {name: torch.empty(shape) for name, shape in {
        "a": (16, 256), "b": (16, 256), "c": (32, 256),
        "dense": (16, 256), "d": (16, 256),
    }.items()}
    pending = [("a", "family", 100), ("a", "family", 200),
               ("b", "family", 100), ("c", "family", 100),
               ("dense", "family", 100), ("d", "family", 100)]
    batches = campaign._anchor_batches(
        pending, weights=weights, expert_members={"a", "b", "c", "d"},
        batch_size=2)
    assert sorted(item for batch in batches for item in batch) == sorted(pending)
    assert [("a", "family", 100), ("b", "family", 100)] in batches
    assert [("dense", "family", 100)] in batches
    assert max(map(len, batches)) == 2
    assert campaign._anchor_batches(pending, weights=weights,
        expert_members=set(weights), batch_size=1) == [[item] for item in pending]


def test_batch_adapter_calls_producer_once_with_separate_activation_inputs(monkeypatch):
    seen = []
    weights = [torch.full((16, 256), float(i)) for i in (1, 2)]
    hessians = [{"ldl": torch.eye(256) * i, "ldl_block": 128} for i in (1, 2)]
    monkeypatch.setattr(render, "_encoder_accepts_hessian",
                        lambda: (True, {"ldl", "ldl_block"}, {}))
    monkeypatch.setattr(render, "rung_accepts_hessian", lambda *_args: True)

    def encode(values, **kwargs):
        seen.append((values, kwargs))
        return [SimpleNamespace(blob=bytes([i])) for i in range(len(values))]

    monkeypatch.setattr(render._tessera_export, "encode_linears", encode)
    from tessera import unit_artifact
    monkeypatch.setattr(unit_artifact, "read_unit_artifact",
                        lambda blob, **_kwargs: weights[blob[0]])
    out = render.encode_tessera_units(weights, FMT, activation_kwargs=hessians)
    assert len(seen) == 1
    assert seen[0][1]["names"] == [FMT, FMT]
    assert seen[0][1]["per_unit"] == hessians
    assert all(torch.equal(actual[0], expected) for actual, expected in zip(out, weights))


def test_batch_missing_hessian_refuses_before_any_producer_call(monkeypatch):
    monkeypatch.setattr(render._tessera_export, "encode_linears",
                        lambda *_args, **_kwargs: pytest.fail("encoded invalid inputs"))
    with pytest.raises(render.HessianContractError, match="no Hessian"):
        render.encode_tessera_units([torch.ones(16, 256)] * 2, FMT,
                                    activation_kwargs=[{"ldl": torch.eye(256)}, None])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real encoder CLI requires CUDA")
def test_real_cli_batch_wire_prices_and_resume_equal_scalar(monkeypatch, tmp_path):
    import pickle
    import test_tessera_campaign_packed as fixture
    monkeypatch.setattr(fixture, "HIDDEN", 256)
    monkeypatch.setattr(fixture, "INTER", 256)
    _bridge_main_fixture = fixture._bridge_main_fixture

    scalar_measure = campaign._measure_anchor
    instance, argv, model, _ = _bridge_main_fixture(monkeypatch, tmp_path)
    model.to("cuda")
    monkeypatch.setattr(instance, "_measure_anchor", scalar_measure)
    argv[argv.index("--hessian") + 1] = "require"
    real_batch = render._tessera_export.encode_linears
    seen = []

    def tracked(weights, **kwargs):
        seen.append(len(weights))
        return real_batch(weights, **kwargs)

    monkeypatch.setattr(render._tessera_export, "encode_linears", tracked)
    assert instance.main(argv) == 0
    baseline = pickle.loads((tmp_path / "cost.pkl").read_bytes())
    baseline_wires = {path.name: path.read_bytes()
                      for path in (tmp_path / "cache" / "wire").glob("*.tessera")}
    batch_args = list(argv)
    batch_args[batch_args.index("--out") + 1] = str(tmp_path / "batch.pkl")
    batch_args[batch_args.index("--cache-dir") + 1] = str(tmp_path / "batch-cache")
    batch_args += ["--anchor-batch-size", "4"]
    seen.clear()
    assert instance.main(batch_args) == 0
    assert 4 in seen
    candidate = pickle.loads((tmp_path / "batch.pkl").read_bytes())
    def numerical_costs(costs):
        return {name: {fmt: {key: value for key, value in row.items()
                            if key not in {"encode_seconds", "encode_seconds_accounting", "encoding_batch_size"}}
                       for fmt, row in rows.items()}
                for name, rows in costs.items()}
    assert numerical_costs(candidate["costs"]) == numerical_costs(baseline["costs"])
    batch_wires = {path.name: path.read_bytes()
                   for path in (tmp_path / "batch-cache" / "wire").glob("*.tessera")}
    assert batch_wires == baseline_wires and batch_wires
    monkeypatch.setattr(render._tessera_export, "encode_linears",
                        lambda *_args, **_kwargs: pytest.fail("resume encoded again"))
    monkeypatch.setattr(render._tessera_export, "encode_linear",
                        lambda *_args, **_kwargs: pytest.fail("resume encoded again"))
    # Batch width is an execution setting, so changing it resumes the same rows.
    batch_args[-1] = "2"
    assert instance.main(batch_args) == 0
    wire = next((tmp_path / "batch-cache" / "wire").glob("*.tessera"))
    blob = bytearray(wire.read_bytes())
    blob[-1] ^= 1
    wire.write_bytes(blob)
    with pytest.raises(RuntimeError, match="cached wire identity refused"):
        instance.main(batch_args)


def test_batch_scoring_keeps_each_units_static_activation_scale(monkeypatch, tmp_path):
    from prismaquant import format_registry as fr
    weights = [torch.ones(16, 256), torch.full((16, 256), 2.0)]
    acts = [torch.ones(2, 256), torch.full((2, 256), 3.0)]
    scales = {"a": 0.5, "b": 2.0}
    spec = SimpleNamespace(
        static_activation_contract=SimpleNamespace(execution="fixture-static",
            quantize_dequantize=lambda x, scale: x * scale),
        bits_for_shape=lambda shape: 8 * shape[0] * shape[1],
        memory_bytes_for_shape=lambda shape: shape[0] * shape[1], act_dtype_name="a8")
    get_format = fr.get_format
    monkeypatch.setattr(fr, "get_format", lambda fmt: spec if fmt == FMT else get_format(fmt))
    monkeypatch.setattr(render, "encode_tessera_units",
        lambda values, *_args, **_kwargs: [(w * 0.75, b"wire") for w in values])
    monkeypatch.setattr(campaign, "_encode_and_render",
        lambda w, *_args, **_kwargs: (w * 0.75, b"wire"))
    cache = SimpleNamespace(weights={}, cache_dir=None)
    wire_dir = tmp_path / "wire"
    wire_dir.mkdir()
    batch = campaign._measure_anchor_batch(qnames=["a", "b"], weights=weights,
        activations=acts, format_name=FMT, cache=cache, wire_dir=wire_dir,
        hessian_required=False, static_input_scales=scales)
    scalar = [campaign._measure_anchor(qname=name, weight=w, activations=x,
        format_name=FMT, cache=cache, wire_dir=wire_dir, hessian_required=False,
        static_input_scale=scales[name]) for name, w, x in zip(scales, weights, acts)]
    assert [row.input_global_scale for row in batch] == [0.5, 2.0]
    assert [row.dloss for row in batch] == [row.dloss for row in scalar]
    assert batch[0].dloss != batch[1].dloss
    assert all(row.encoding_batch_size == 2 for row in batch)


def test_requested_batch_refuses_old_producer_before_loading_model(monkeypatch, tmp_path):
    monkeypatch.setattr(render._tessera_export, "encode_linears", None)
    with pytest.raises(RuntimeError, match="no encode_linears API"):
        campaign.main(["--model", "must-not-load", "--out", str(tmp_path / "cost.pkl"),
                       "--cache-dir", str(tmp_path / "cache"), "--anchor-batch-size", "4"])
