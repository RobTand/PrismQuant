"""Executable contracts for the value-bearing trellis render/cache seam."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from prismaquant.production_weight_cache import (
    ProductionWeightCache,
    _cache_pair_identity_filename,
    _cache_weight_filename,
    _source_weight_value_identity,
    render_production_weight,
)
from prismaquant.trellis_encoder import encoder_source_sha256
from prismaquant.trellis_footprint import trellis_tensor_payload_breakdown
from prismaquant.trellis_formats import (
    E2M1_FAMILY,
    E4M3FN_NAN_CODES,
    E4M3_FAMILY,
    native_code_value,
)
from prismaquant.trellis_render import (
    EXECUTED_ACTIVATION_CONTRACT,
    TRELLIS_ENCODE_PLAN_SET_SCHEMA,
    TrellisEncodePlan,
    TrellisRenderError,
    TrellisRenderRecipe,
    TrellisWireSink,
    TrellisWireSinkError,
    build_trellis_pair_identity,
    load_trellis_encode_plan_set,
    trellis_tensor_value_sha256,
)
from prismaquant.trellis_wire import (
    TrellisWire,
    decode_values_torch,
    pack_planes,
)


_E2_CODES_R2 = (15, 13, 11, 9, 8, 2, 4, 7)
_H = "0" * 64


def _e2_alphabet(rate: int) -> tuple[int, ...]:
    ordered = tuple(sorted(
        range(16), key=lambda code: (native_code_value(E2M1_FAMILY, code), code)
    ))
    count = 1 << (rate + 1)
    if rate == 2:
        return _E2_CODES_R2
    return tuple(
        ordered[index * (len(ordered) - 1) // (count - 1)]
        for index in range(count)
    )


def _e4_alphabet(rate: int) -> tuple[int, ...]:
    ordered = tuple(sorted(
        (code for code in range(256) if code not in E4M3FN_NAN_CODES),
        key=lambda code: (native_code_value(E4M3_FAMILY, code), code),
    ))
    count = 1 << (rate + 1)
    return tuple(
        ordered[index * (len(ordered) - 1) // (count - 1)]
        for index in range(count)
    )


def _plan(
    *,
    family: str = E2M1_FAMILY,
    rows: int = 1,
    source_sha256: str | None = None,
    measured_activation_contract: str | None = None,
    activation_input_global_scale: float | None | object = ...,
    schedule: tuple[int, ...] | None = None,
) -> tuple[TrellisEncodePlan, torch.Tensor]:
    if family == E2M1_FAMILY:
        body_rate_q256 = 512
        schedule = schedule or (2,) * 256
        alphabets = {
            rate: _e2_alphabet(rate)
            for rate in sorted(set(schedule))
            if rate < 4
        }
        scale_rule = "static_6"
        activation_scale = (
            4.0 if activation_input_global_scale is ...
            else activation_input_global_scale
        )
    else:
        body_rate_q256 = 1024
        schedule = schedule or (4,) * 256
        alphabets = {
            rate: _e4_alphabet(rate)
            for rate in sorted(set(schedule))
            if rate < 8
        }
        scale_rule = "row_fp32_amax_448"
        activation_scale = (
            None if activation_input_global_scale is ...
            else activation_input_global_scale
        )
    shape = (rows, 256)
    footprint = trellis_tensor_payload_breakdown(
        shape,
        family=family,
        body_rate_q256=body_rate_q256,
        layout="tight_offsets",
        schedule=schedule,
        alphabets=alphabets,
    )
    recipe = TrellisRenderRecipe(
        family=family,
        body_rate_q256=body_rate_q256,
        layout="tight_offsets",
        schedule_identity_sha256=footprint["schedule_identity_sha256"],
        alphabet_identity_sha256=footprint["alphabet_identity_sha256"],
        pre_render_recipe_identity_sha256=footprint[
            "pre_render_recipe_identity_sha256"
        ],
        encoder_source_sha256=(source_sha256 or encoder_source_sha256()),
        encoder_sb_chunk=max(1, rows),
        encoder_determinism_mode="on",
        encoder_tailbite_candidates=4,
        encoder_backend="eager",
        encoder_point_route="full",
        scale_rule=scale_rule,
    )
    col_weights = torch.linspace(0.5, 1.5, 256, dtype=torch.float32)
    plan = TrellisEncodePlan(
        shape=shape,
        schedule=schedule,
        alphabets=alphabets,
        priced_footprint=footprint,
        recipe=recipe,
        col_weights_sha256=trellis_tensor_value_sha256(col_weights),
        measured_activation_contract=(
            recipe.activation_contract
            if measured_activation_contract is None
            else measured_activation_contract
        ),
        activation_input_global_scale=activation_scale,  # type: ignore[arg-type]
    )
    return plan, col_weights


@pytest.fixture(scope="module")
def e2_case():
    plan, col_weights = _plan()
    torch.manual_seed(20260830)
    weight = torch.randn(plan.shape, dtype=torch.bfloat16)
    return plan, col_weights, weight


@pytest.fixture(scope="module")
def e2_rendered(e2_case):
    plan, col_weights, weight = e2_case
    sink = TrellisWireSink(qname="layer", fmt=plan.fmt, plan=plan)
    decoded = render_production_weight(
        weight,
        plan.fmt,
        qname="layer",
        activations={"layer": torch.randn(2, 256)},
        levers={"gptq": False, "weighted_vq": True},
        col_weights=col_weights,
        trellis_wire_out=sink,
    )
    return plan, col_weights, weight, sink, decoded


def _pair_identity(plan, col_weights, weight, *, qname="layer"):
    shape, source_sha = _source_weight_value_identity(weight)
    return build_trellis_pair_identity(
        qname=qname,
        fmt=plan.fmt,
        shape=shape,
        recipe=plan.recipe,
        source_weight_sha256=source_sha,
        source_weight_dtype=str(weight.dtype),
        col_weights_sha256=trellis_tensor_value_sha256(col_weights),
        col_weights_shape=tuple(col_weights.shape),
        activation_input_global_scale=plan.activation_input_global_scale,
        calibration_hash="calibration-fixture",
        git_commit="d" * 40,
        producer_source_sha256="e" * 64,
    )


def _rendered_cache_case(
    family: str,
) -> tuple[TrellisEncodePlan, ProductionWeightCache]:
    # Four rows crosses iter_quantizable_tensors' real production minimum
    # (1,000 parameters), so the activation hook resolves this exactly as it
    # resolves a model Linear rather than through a test-only shortcut.
    plan, col_weights = _plan(family=family, rows=4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260830 if family == E2M1_FAMILY else 20260831)
    weight = torch.randn(
        plan.shape, dtype=torch.bfloat16, generator=generator
    )
    sink = TrellisWireSink(qname="layer", fmt=plan.fmt, plan=plan)
    decoded = render_production_weight(
        weight,
        plan.fmt,
        qname="layer",
        activations={"layer": torch.randn(2, 256, generator=generator)},
        levers={"gptq": False, "weighted_vq": True},
        col_weights=col_weights,
        trellis_wire_out=sink,
    )
    cache = ProductionWeightCache(weights={}, levers={}, metadata={})
    cache.store_trellis_render(
        qname="layer",
        fmt=plan.fmt,
        sink=sink,
        decoded_tensor=decoded,
        identity=_pair_identity(plan, col_weights, weight),
        render_score={"score": 1.0, "metric": "fixture"},
    )
    return plan, cache


def test_trellis_render_requires_a_wire_sink(e2_case):
    plan, col_weights, weight = e2_case
    with pytest.raises(ValueError, match="requires an explicit wire sink"):
        render_production_weight(
            weight,
            plan.fmt,
            qname="layer",
            activations={},
            levers={"weighted_vq": True},
            col_weights=col_weights,
        )


def test_a_wire_sink_on_a_non_trellis_format_is_refused(e2_case):
    plan, _, weight = e2_case
    sink = TrellisWireSink(qname="layer", fmt=plan.fmt, plan=plan)
    with pytest.raises(ValueError, match="non-trellis format"):
        render_production_weight(
            weight,
            "NVFP4",
            qname="layer",
            activations={},
            levers={"gptq": False},
            trellis_wire_out=sink,
        )


def test_render_blob_matches_priced_wire_and_independent_decode(e2_rendered):
    plan, _, _, sink, decoded = e2_rendered
    assert sink.filled
    assert len(sink.blob) == plan.expected_wire_bytes
    wire = TrellisWire.from_bytes(sink.blob)
    assert wire.to_bytes() == sink.blob
    assert wire.schedule == plan.schedule
    assert dict(wire.alphabets) == dict(plan.alphabets)
    assert torch.equal(
        decoded,
        decode_values_torch(sink.blob, dtype=decoded.dtype),
    )


def test_e4m3_family_renders_a_real_wire():
    plan, col_weights = _plan(family=E4M3_FAMILY)
    torch.manual_seed(20260831)
    weight = torch.randn(plan.shape, dtype=torch.bfloat16)
    sink = TrellisWireSink(qname="layer", fmt=plan.fmt, plan=plan)
    decoded = render_production_weight(
        weight,
        plan.fmt,
        qname="layer",
        activations={"layer": torch.randn(2, 256)},
        levers={"weighted_vq": True},
        col_weights=col_weights,
        trellis_wire_out=sink,
    )
    wire = TrellisWire.from_bytes(sink.blob)
    assert wire.family == E4M3_FAMILY
    assert wire.global_scale_real == 1.0
    assert torch.equal(decoded, decode_values_torch(sink.blob, dtype=decoded.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("family", (E2M1_FAMILY, E4M3_FAMILY))
def test_triton_encoder_matches_eager_primary_wire_on_cuda(family):
    from prismaquant.trellis_encoder import encode_trellis_planes

    plan, col_weights = _plan(family=family, rows=2)
    torch.manual_seed(20260832)
    weight = torch.randn(plan.shape, device="cuda", dtype=torch.float32)
    vector = col_weights.to("cuda")
    common = dict(
        family=family,
        schedule=plan.schedule,
        alphabets=plan.alphabets,
        scale_rule=plan.recipe.scale_rule,
        sb_chunk=2,
        determinism_mode="on",
        tailbite_candidates=4,
        point_route="full",
    )
    encoded = {
        backend: encode_trellis_planes(
            weight, vector, backend=backend, **common
        )
        for backend in ("eager", "triton")
    }
    blobs = {}
    for backend, result in encoded.items():
        blobs[backend] = pack_planes(
            family=family,
            body_rate_q256=plan.recipe.body_rate_q256,
            schedule=plan.schedule,
            layout=plan.recipe.layout,
            u_bits=result.u_bits,
            point_indices=result.point_indices,
            bypass_codes=result.bypass_codes,
            alphabets=plan.alphabets,
            scale_blob=result.scale_blob,
            global_scale_real=result.global_scale_real,
        ).to_bytes()
    assert blobs["triton"] == blobs["eager"]
    assert torch.equal(
        encoded["triton"].reconstruction,
        encoded["eager"].reconstruction,
    )


def test_trellis_render_requires_exact_weighted_inputs(e2_case):
    plan, col_weights, weight = e2_case
    sink = TrellisWireSink(qname="layer", fmt=plan.fmt, plan=plan)
    with pytest.raises(TrellisRenderError, match="weighted_vq cannot be disabled"):
        render_production_weight(
            weight,
            plan.fmt,
            qname="layer",
            activations={},
            levers={"weighted_vq": False},
            col_weights=col_weights,
            trellis_wire_out=sink,
        )
    with pytest.raises(TrellisRenderError, match="col_weights value differs"):
        render_production_weight(
            weight,
            plan.fmt,
            qname="layer",
            activations={},
            levers={"weighted_vq": True},
            col_weights=col_weights + 1,
            trellis_wire_out=TrellisWireSink(
                qname="layer", fmt=plan.fmt, plan=plan
            ),
        )


def test_render_revalidates_mutable_plan_values_before_encoding(e2_case):
    plan, col_weights, weight = e2_case
    mutable = TrellisEncodePlan.from_mapping(plan.as_dict())
    assert isinstance(mutable.alphabets, dict)
    mutable.alphabets[2] = tuple(reversed(mutable.alphabets[2]))
    sink = TrellisWireSink(qname="layer", fmt=mutable.fmt, plan=mutable)
    with pytest.raises(TrellisRenderError, match="trust-boundary revalidation"):
        render_production_weight(
            weight,
            mutable.fmt,
            qname="layer",
            activations={"layer": torch.randn(2, 256)},
            levers={"gptq": False, "weighted_vq": True},
            col_weights=col_weights,
            trellis_wire_out=sink,
        )


def test_encoder_source_and_joint_nvfp4_semantics_fail_before_encode(e2_case):
    _, col_weights, weight = e2_case
    stale, _ = _plan(source_sha256=_H)
    with pytest.raises(TrellisRenderError, match="encoder source digest differs"):
        render_production_weight(
            weight, stale.fmt, qname="layer", activations={},
            levers={"weighted_vq": True}, col_weights=col_weights,
            trellis_wire_out=TrellisWireSink(
                qname="layer", fmt=stale.fmt, plan=stale
            ),
        )
    plan, _ = _plan()
    with pytest.raises(TrellisRenderError, match="joint_global_real"):
        render_production_weight(
            weight, plan.fmt, qname="layer", activations={},
            levers={"weighted_vq": True}, col_weights=col_weights,
            joint_global_real=torch.tensor(1.0),
            trellis_wire_out=TrellisWireSink(
                qname="layer", fmt=plan.fmt, plan=plan
            ),
        )


def test_plan_refuses_w_a16_and_missing_or_spurious_a_side_scale():
    with pytest.raises(TrellisRenderError, match="anchor activation contract"):
        _plan(measured_activation_contract="bf16")
    with pytest.raises(TrellisRenderError, match="activation_input_global_scale"):
        _plan(activation_input_global_scale=None)
    with pytest.raises(TrellisRenderError, match="per-token dynamic"):
        _plan(family=E4M3_FAMILY, activation_input_global_scale=4.0)


def test_sink_is_single_assignment_and_checks_priced_length(e2_case):
    plan, _, _ = e2_case
    sink = TrellisWireSink(qname="layer", fmt=plan.fmt, plan=plan)
    with pytest.raises(TrellisWireSinkError, match="priced at"):
        sink.accept(b"\0" * (plan.expected_wire_bytes - 1), recipe=plan.recipe)
    sink.accept(b"\0" * plan.expected_wire_bytes, recipe=plan.recipe)
    with pytest.raises(TrellisWireSinkError, match="already holds a blob"):
        sink.accept(b"\0" * plan.expected_wire_bytes, recipe=plan.recipe)


def test_pair_identity_disambiguates_one_format_key(e2_case):
    plan, col_weights, weight = e2_case
    first = _pair_identity(plan, col_weights, weight)
    altered_recipe = TrellisRenderRecipe(**{
        **plan.recipe.__dict__,
        "schedule_identity_sha256": "1" * 64,
    })
    shape, source_sha = _source_weight_value_identity(weight)
    second = build_trellis_pair_identity(
        qname="layer",
        fmt=plan.fmt,
        shape=shape,
        recipe=altered_recipe,
        source_weight_sha256=source_sha,
        source_weight_dtype=str(weight.dtype),
        col_weights_sha256=trellis_tensor_value_sha256(col_weights),
        col_weights_shape=tuple(col_weights.shape),
        activation_input_global_scale=plan.activation_input_global_scale,
        calibration_hash="calibration-fixture",
        git_commit="d" * 40,
        producer_source_sha256="e" * 64,
    )
    assert first["qname"] == second["qname"]
    assert first["format"] == second["format"]
    assert first["recipe_identity_sha256"] != second["recipe_identity_sha256"]


def test_wire_blob_is_primary_across_store_load_and_lru(
    e2_rendered, tmp_path: Path,
):
    plan, col_weights, weight, sink, decoded = e2_rendered
    identity = _pair_identity(plan, col_weights, weight)
    cache = ProductionWeightCache(
        weights={}, levers={}, cache_dir=str(tmp_path), metadata={}
    )
    record = cache.store_trellis_render(
        qname="layer",
        fmt=plan.fmt,
        sink=sink,
        decoded_tensor=decoded,
        identity=identity,
        render_score={"score": 1.25, "metric": "fixture"},
    )
    shard = torch.load(
        tmp_path / _cache_weight_filename("layer", plan.fmt),
        map_location="cpu",
        weights_only=True,
    )
    assert shard.dtype == torch.uint8 and shard.ndim == 1
    assert shard.numpy().tobytes() == sink.blob
    assert cache.estimate_nbytes() == (
        len(sink.blob) + decoded.numel() * decoded.element_size()
    )
    assert cache.get_wire_blob("layer", plan.fmt) == sink.blob
    assert torch.equal(cache.get("layer", plan.fmt), decoded)
    assert record["rendered_wire_identity_sha256"] == hashlib.sha256(
        sink.blob
    ).hexdigest()

    cache.compact_for_pickle()
    reloaded = ProductionWeightCache(
        weights=dict(cache.weights),
        levers={},
        cache_dir=str(tmp_path),
        metadata=copy.deepcopy(cache.metadata),
    )
    assert reloaded.get_wire_blob("layer", plan.fmt) == sink.blob
    assert torch.equal(reloaded.get("layer", plan.fmt), decoded)


def test_dense_fill_publishes_and_resumes_the_same_wire_without_reencoding(
    tmp_path: Path, monkeypatch,
):
    from torch import nn
    from prismaquant.production_weight_cache import fill_production_weight_cache

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 256)
            self.layer = nn.Linear(256, 4, bias=False, dtype=torch.bfloat16)

        def forward(self, ids, use_cache=False):
            return self.layer(self.embed(ids).to(torch.bfloat16))

    plan, col_weights = _plan(rows=4)
    model = TinyModel()
    calib_ids = torch.tensor([[1, 2]], dtype=torch.long)
    fill_kwargs = dict(
        formats=(plan.fmt,),
        render_assignment={"layer": plan.fmt},
        levers={
            "gptq": False,
            "static_act_order": False,
            "joint_scale_opt": False,
            "scale_sweep": False,
            "weighted_vq": True,
        },
        progress=False,
        cache_dir=tmp_path,
        col_weights={"layer": col_weights},
        trellis_plans={("layer", plan.fmt): plan},
    )
    cache = fill_production_weight_cache(
        model,
        calib_ids,
        ["layer"],
        **fill_kwargs,
    )
    assert not cache.failed
    blob = cache.get_wire_blob("layer", plan.fmt)
    assert blob is not None and len(blob) == plan.expected_wire_bytes
    shard = torch.load(
        tmp_path / _cache_weight_filename("layer", plan.fmt),
        map_location="cpu", weights_only=True,
    )
    assert shard.dtype == torch.uint8 and shard.numpy().tobytes() == blob
    pair = cache.metadata["trellis_cache_pair_artifacts"]["records"][
        f"layer|{plan.fmt}"
    ]
    assert pair["rendered_wire_identity_sha256"] == hashlib.sha256(blob).hexdigest()
    assert pair["identity"]["activation_contract"] == plan.recipe.activation_contract
    score = cache.metadata["render_scores"]["records"][f"layer|{plan.fmt}"]
    assert score["activation_quantized"] is True

    import prismaquant.trellis_encoder as encoder

    def _reencode_is_a_bug(*args, **kwargs):
        raise AssertionError("resume re-encoded a primary trellis wire")

    monkeypatch.setattr(encoder, "encode_trellis_planes", _reencode_is_a_bug)
    resumed = fill_production_weight_cache(
        model, calib_ids, ["layer"], **fill_kwargs
    )
    assert resumed.get_wire_blob("layer", plan.fmt) == blob


def test_cache_rejects_decoded_dtype_outside_source_narrowing_contract(
    e2_rendered, tmp_path: Path,
):
    plan, col_weights, weight, sink, decoded = e2_rendered
    identity = _pair_identity(plan, col_weights, weight)
    cache = ProductionWeightCache(
        weights={}, levers={}, cache_dir=str(tmp_path), metadata={}
    )
    cache.store_trellis_render(
        qname="layer", fmt=plan.fmt, sink=sink, decoded_tensor=decoded,
        identity=identity, render_score={"score": 1.0},
    )
    sidecar_path = tmp_path / _cache_pair_identity_filename("layer", plan.fmt)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["decoded"]["dtype"] = "torch.float32"
    sidecar_path.write_text(json.dumps(sidecar))
    reloaded = ProductionWeightCache(
        weights={("layer", plan.fmt): _cache_weight_filename("layer", plan.fmt)},
        levers={}, cache_dir=str(tmp_path), metadata={},
    )
    with pytest.raises(RuntimeError, match="cache narrowing contract"):
        reloaded.get_wire_blob("layer", plan.fmt)


def test_cache_rejects_a_wire_whose_hash_was_updated_but_recipe_was_not(
    e2_rendered, tmp_path: Path,
):
    plan, col_weights, weight, sink, decoded = e2_rendered
    identity = _pair_identity(plan, col_weights, weight)
    cache = ProductionWeightCache(
        weights={}, levers={}, cache_dir=str(tmp_path), metadata={}
    )
    cache.store_trellis_render(
        qname="layer", fmt=plan.fmt, sink=sink, decoded_tensor=decoded,
        identity=identity, render_score={"score": 1.0},
    )

    alternate_schedule = tuple([1, 3] * 128)
    alternate = pack_planes(
        family=E2M1_FAMILY,
        body_rate_q256=512,
        schedule=alternate_schedule,
        layout="tight_offsets",
        u_bits=torch.zeros(1, 256, dtype=torch.uint8),
        point_indices=torch.zeros(1, 256, dtype=torch.uint8),
        bypass_codes=torch.zeros(1, 256, dtype=torch.uint8),
        alphabets={1: _e2_alphabet(1), 3: _e2_alphabet(3)},
        scale_blob=bytes([0x38] * 16),
        global_scale_real=1.0,
    ).to_bytes()
    shard_path = tmp_path / _cache_weight_filename("layer", plan.fmt)
    torch.save(torch.frombuffer(bytearray(alternate), dtype=torch.uint8), shard_path)
    sidecar_path = tmp_path / _cache_pair_identity_filename("layer", plan.fmt)
    sidecar = json.loads(sidecar_path.read_text())
    digest = hashlib.sha256(alternate).hexdigest()
    sidecar["rendered_wire_identity_sha256"] = digest
    sidecar["wire_bytes"] = len(alternate)
    sidecar["wire"] = {
        "shape": [len(alternate)],
        "dtype": "torch.uint8",
        "logical_bytes": len(alternate),
        "content_sha256": digest,
    }
    sidecar_path.write_text(json.dumps(sidecar))
    reloaded = ProductionWeightCache(
        weights={("layer", plan.fmt): shard_path.name},
        levers={}, cache_dir=str(tmp_path), metadata={},
    )
    with pytest.raises(RuntimeError, match="wire recipe differs"):
        reloaded.get_wire_blob("layer", plan.fmt)


def test_plan_set_round_trips_without_ambient_lookup(e2_case, tmp_path: Path):
    plan, _, _ = e2_case
    path = tmp_path / "plans.json"
    path.write_text(json.dumps({
        "schema": TRELLIS_ENCODE_PLAN_SET_SCHEMA,
        "records": [{"qname": "layer", **plan.as_dict()}],
    }))
    loaded = load_trellis_encode_plan_set(path)
    assert loaded[("layer", plan.fmt)] == plan


def test_recipe_identity_binds_chunk_backend_route_and_determinism(e2_case):
    plan, _, _ = e2_case
    base = plan.recipe.__dict__
    for field, value in (
        ("encoder_sb_chunk", 2),
        ("encoder_backend", "triton"),
        ("encoder_point_route", "windowed"),
        ("encoder_determinism_mode", "off"),
    ):
        changed = TrellisRenderRecipe(**{**base, field: value})
        assert changed.identity_sha256 != plan.recipe.identity_sha256


def test_executed_activation_contracts_are_native_a_equals_w():
    assert EXECUTED_ACTIVATION_CONTRACT == {
        E2M1_FAMILY: "e2m1_group16_ue4m3_static",
        E4M3_FAMILY: "fp8_per_token_dynamic",
    }


@pytest.mark.parametrize("family", (E2M1_FAMILY, E4M3_FAMILY))
def test_kl_activation_hook_applies_authenticated_native_a_side(family):
    from torch import nn
    from prismaquant import format_registry as fr
    from prismaquant.validate_assignments_kl import _TrellisActivationHooks

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(
                256, 4, bias=False, dtype=torch.bfloat16
            )

        def forward(self, value):
            return self.layer(value)

    plan, cache = _rendered_cache_case(family)
    model = TinyModel()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(101)
    value = (
        torch.randn(2, 256, generator=generator) * 3.7
    ).to(torch.bfloat16)
    identity = cache.get_trellis_activation_identity("layer", plan.fmt)
    if family == E2M1_FAMILY:
        from prismaquant.nvfp4_activation_contract import (
            nvfp4_activation_qdq_served,
        )
        expected = nvfp4_activation_qdq_served(
            value, float(identity["activation_input_global_scale"])
        )
    else:
        expected = fr.get_format(
            "FP8_E4M3"
        ).activation_quantize_dequantize(value)

    hooks = _TrellisActivationHooks(
        model, {"layer": plan.fmt}, cache, profile=None
    )
    seen: list[torch.Tensor] = []
    hooks.install()
    capture = model.layer.register_forward_pre_hook(
        lambda _module, args, _kwargs: seen.append(args[0].detach().clone()),
        with_kwargs=True,
    )
    try:
        model(value)
    finally:
        capture.remove()
        hooks.remove()
    assert len(seen) == 1
    assert torch.equal(seen[0], expected)
    assert not torch.equal(seen[0], value)
    assert hooks.summary()["calls"] == {"layer": 1}
    assert hooks.summary()["contracts"] == {
        "layer": plan.recipe.activation_contract
    }


def test_kl_activation_hook_refuses_tampered_contract_before_forward():
    from torch import nn
    from prismaquant.validate_assignments_kl import _TrellisActivationHooks

    plan, cache = _rendered_cache_case(E2M1_FAMILY)
    record = cache.metadata["trellis_cache_pair_artifacts"]["records"][
        f"layer|{plan.fmt}"
    ]
    record["identity"]["activation_contract"] = "bf16"
    model = nn.Module()
    model.layer = nn.Linear(256, 4, bias=False, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="activation contract differs"):
        _TrellisActivationHooks(
            model, {"layer": plan.fmt}, cache, profile=None
        )


def test_registry_assignment_partition_never_rounds_trellis_to_bf16():
    from prismaquant.validate_assignments_kl import (
        _activation_quant_assignment,
    )

    assert _activation_quant_assignment({
        "trellis": "TCQ_E2M1_R512",
        "normal": "NVFP4",
    }) == {"normal": "NVFP4"}


def test_inplace_kl_executes_trellis_a_equals_w_hook(tmp_path: Path):
    from types import SimpleNamespace
    from torch import nn
    import torch.nn.functional as F
    from prismaquant.build_rtn_cache import kl_divergence
    from prismaquant.nvfp4_activation_contract import (
        nvfp4_activation_qdq_served,
    )
    from prismaquant.validate_assignments_kl import (
        _measure_inplace_assignment_kl,
    )

    class TinyCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 256, dtype=torch.bfloat16)
            self.layer = nn.Linear(
                256, 4, bias=False, dtype=torch.bfloat16
            )

        def forward(self, token_ids):
            return SimpleNamespace(logits=self.layer(self.embed(token_ids)))

    plan, cache = _rendered_cache_case(E2M1_FAMILY)
    model = TinyCausalLM()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(321)
    with torch.no_grad():
        model.embed.weight.copy_(
            (torch.randn(8, 256, generator=generator) * 3.7).to(
                torch.bfloat16
            )
        )
    calib_ids = torch.tensor([[1, 2]], dtype=torch.long)
    rendered = cache.get("layer", plan.fmt)
    assert rendered is not None
    hidden = model.embed(calib_ids)
    quantized_hidden = nvfp4_activation_qdq_served(
        hidden, float(plan.activation_input_global_scale)
    )
    native_logits = F.linear(quantized_hidden, rendered)
    a16_logits = F.linear(hidden, rendered)
    teacher = F.log_softmax(native_logits.float(), dim=-1)
    assert float(kl_divergence(a16_logits, teacher).item()) > 1e-6

    mean, values, stats = _measure_inplace_assignment_kl(
        model,
        {"layer": plan.fmt},
        calib_ids,
        [teacher],
        work_root=tmp_path,
        profile=None,
        production_cache=cache,
        kl_scope="full_sequence",
        use_cuda_graphs=False,
    )
    assert abs(mean) < 1e-7
    assert len(values) == 1 and abs(values[0]) < 1e-7
    assert stats["activation_hooks"]["trellis"] == {
        "plans": 1,
        "calls": {"layer": 1},
        "contracts": {"layer": plan.recipe.activation_contract},
        "formats": {"layer": plan.fmt},
    }


_PRE_CHANGE_RENDER_SHA256 = {
    "BF16": "6c7a110784ebed6d9b1ea72fb46d1494c5144d68306dfd0f77d4d39a1a931c3f",
    "FP8_E4M3": "23ae708a2781234c09578a8beb09d6104dc153da2ae8dba2f991c4fdb1f2efb0",
    "MXFP4": "f8e64c110e16a4c9e4b6c33f66cbd58b887c01ba8d83c353dc59f1267d8c53ef",
    "NVFP4": "cdeda44496e21512642a2762a6b6055f4f35d2e39a990b898e4c27344a889aed",
}


@pytest.mark.parametrize("fmt", tuple(_PRE_CHANGE_RENDER_SHA256))
def test_every_existing_render_path_is_byte_identical_to_pre_change(fmt):
    """Hashes were captured from e9c30a6 before the codec/cache implementation."""
    levers = {
        "gptq": False,
        "static_act_order": False,
        "joint_scale_opt": False,
        "scale_sweep": False,
    }
    torch.manual_seed(20260830)
    weight = torch.randn(64, 256, dtype=torch.bfloat16)
    before_shape = render_production_weight(
        weight, fmt, qname="layer", activations={}, levers=levers,
    )
    after_shape = render_production_weight(
        weight, fmt, qname="layer", activations={}, levers=levers,
        trellis_wire_out=None,
    )
    before_bytes = before_shape.contiguous().view(torch.uint8).numpy().tobytes()
    after_bytes = after_shape.contiguous().view(torch.uint8).numpy().tobytes()
    assert after_bytes == before_bytes
    assert hashlib.sha256(after_bytes).hexdigest() == _PRE_CHANGE_RENDER_SHA256[fmt]


def test_native_compressed_refusal_names_wrong_container_and_unattested():
    from prismaquant.export_native_compressed import (
        _coerce_runtime_legal_assignment,
    )

    with pytest.raises(ValueError) as excinfo:
        _coerce_runtime_legal_assignment(
            "/does/not/need/to/exist", {"layer": "TCQ_E2M1_R512"}
        )
    message = str(excinfo.value)
    assert "not compressed-tensors" in message
    assert "route_status=unattested" in message
    assert "can render and cache" in message
