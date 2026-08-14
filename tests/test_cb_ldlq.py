"""Production-contract tests for flag-gated CB LDLQ assignment."""
from __future__ import annotations

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant.cb_ldlq import fill_empty_expert_activation_rows
from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_warm_state import CBWarmStateStore, build_warm_record
from prismaquant.measure_quant_cost import (
    _cb_cost_quantize_dequantize,
    cost_payload_provenance,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_serialization_stamps,
    cb_fields_for_context,
    cb_serialization_context_from_env,
    cb_serialization_context_stamp,
    validate_cb_assignment_serialization_stamps,
    validate_cb_serialization_context_stamp,
)


def _case(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(8, 256, generator=generator) * 0.25
    latent = torch.randn(48, 64, generator=generator)
    activations = torch.cat(
        (
            latent,
            0.9 * latent + 0.1 * torch.randn(48, 64, generator=generator),
            -0.8 * latent + 0.2 * torch.randn(48, 64, generator=generator),
            0.7 * latent + 0.3 * torch.randn(48, 64, generator=generator),
        ),
        dim=1,
    )
    return weight, activations, activations.square().mean(dim=0)


def _set_context_env(monkeypatch, *, ldlq: bool) -> None:
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("CB_SCALE_SWEEP", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "1" if ldlq else "0")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "fast")


def test_empty_expert_rows_use_same_layer_routed_pool():
    first = torch.tensor([[1.0, 2.0]])
    third = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    rows, missing = fill_empty_expert_activation_rows(
        (first, torch.empty(0, 2), third),
        qname="model.layers.0.mlp.experts.down_proj",
    )
    assert missing == (1,)
    assert torch.equal(rows[0], first)
    assert torch.equal(rows[2], third)
    assert torch.equal(rows[1], torch.cat((first, third)))


def test_flag_off_is_byte_identical_to_the_existing_encoder(monkeypatch):
    weight, _activations, col_weights = _case()
    spec = fr.get_format("NVFP4_CB_K12")
    monkeypatch.delenv("PRISMAQUANT_CB_LDLQ", raising=False)
    assert cb_serialization_context_from_env().ldlq is False
    _set_context_env(monkeypatch, ldlq=False)
    context = cb_serialization_context_from_env(require_explicit=True)

    baseline, _ = cb.nvfp4_cb_pack(
        weight,
        12,
        grid="fp4",
        mode="product",
        col_weights=col_weights,
        scale_coding="two_tier",
        encode_tier="fast",
    )
    fields = cb_fields_for_context(
        spec,
        weight,
        context=context,
        col_weights=col_weights,
    )
    flagged_off = cb.nvfp4_cb_assemble_bytes(
        fields, 12, grid="fp4", mode="product"
    )

    assert context.ldlq is False
    assert torch.equal(flagged_off, baseline)


def test_flag_on_cost_render_is_deterministic_and_preserves_fitted_scales(
    monkeypatch,
):
    weight, activations, col_weights = _case(3)
    spec = fr.get_format("NVFP4_CB_K12")
    _set_context_env(monkeypatch, ldlq=True)

    first = _cb_cost_quantize_dequantize(
        spec,
        weight,
        col_weights=col_weights,
        activation_rows=activations,
    )
    second = _cb_cost_quantize_dequantize(
        spec,
        weight,
        col_weights=col_weights,
        activation_rows=activations,
    )
    plain_fields = cb.nvfp4_cb_fields(
        weight,
        12,
        col_weights=col_weights,
        scale_coding="two_tier",
        encode_tier="fast",
    )
    ldlq_fields = cb_fields_for_context(
        spec,
        weight,
        context=cb_serialization_context_from_env(require_explicit=True),
        col_weights=col_weights,
        activation_rows=activations,
    )

    assert torch.equal(first, second)
    assert torch.equal(ldlq_fields["scales"], plain_fields["scales"])
    assert torch.equal(ldlq_fields["scale_super"], plain_fields["scale_super"])
    assert torch.equal(ldlq_fields["scale_sub"], plain_fields["scale_sub"])


def test_cost_provenance_carries_ldlq_and_export_refuses_mismatch(monkeypatch):
    spec = fr.get_format("NVFP4_CB_K12")
    plain = CBSerializationContext.production(encode_tier="fast", ldlq=False)
    feedback = CBSerializationContext.production(encode_tier="fast", ldlq=True)
    _set_context_env(monkeypatch, ldlq=True)
    provenance = cost_payload_provenance([spec])

    assert provenance["cb_serialized_payload"]["ldlq"] is True
    try:
        validate_cb_serialization_context_stamp(
            cb_serialization_context_stamp(plain),
            feedback,
            where="export_nvfp4_cb test",
        )
    except ValueError as exc:
        assert "differs from allocator recipe" in str(exc)
        assert "'ldlq': False" in str(exc)
    else:
        raise AssertionError("export accepted a plain/LDLQ context mismatch")

    assignment = {"layers.0.proj": spec.name}
    shapes = {"layers.0.proj": (8, 256)}
    plain_stamps = cb_assignment_serialization_stamps(
        assignment,
        shapes,
        context=plain,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_cb_assignment_serialization_stamps(
            assignment,
            shapes,
            context=feedback,
            stamps=plain_stamps,
            where="export_nvfp4_cb test",
        )


def test_non_ldlq_warm_record_cold_falls_back_under_ldlq(tmp_path):
    weight, _activations, col_weights = _case(5)
    fmt = "NVFP4_CB_K12"
    plain = CBSerializationContext.production(encode_tier="fast", ldlq=False)
    feedback = CBSerializationContext.production(encode_tier="fast", ldlq=True)
    fields = cb.nvfp4_cb_fields(
        weight,
        12,
        col_weights=col_weights,
        scale_coding="two_tier",
        encode_tier="fast",
    )
    store = CBWarmStateStore(tmp_path)
    record = build_warm_record(
        qname="layers.0.proj",
        format_name=fmt,
        source_weight=weight,
        col_weights=col_weights,
        context=plain,
        fields=fields,
    )
    store.write(record)

    loaded = store.load_matching(
        qname="layers.0.proj",
        format_name=fmt,
        source_shape=list(weight.shape),
        source_digest=str(record.metadata["source_digest"]),
        col_weights_shape=list(col_weights.shape),
        col_weights_digest=str(record.metadata["col_weights_digest"]),
        context=feedback,
    )

    assert loaded is None


def test_ldlq_beats_plain_on_correlated_known_better_case():
    # The pilot's known-better construction couples later columns to earlier
    # rounding errors. Repeat that pattern at the production 64-column tile.
    weight, activations, col_weights = _case(0)
    spec = fr.get_format("NVFP4_CB_K12")
    plain = cb_fields_for_context(
        spec,
        weight,
        context=CBSerializationContext.production(encode_tier="fast"),
        col_weights=col_weights,
    )
    feedback = cb_fields_for_context(
        spec,
        weight,
        context=CBSerializationContext.production(
            encode_tier="fast", ldlq=True
        ),
        col_weights=col_weights,
        activation_rows=activations,
    )
    plain_weight = cb.nvfp4_cb_reconstruct(
        plain, 12, grid="fp4", mode="product"
    )
    feedback_weight = cb.nvfp4_cb_reconstruct(
        feedback, 12, grid="fp4", mode="product"
    )
    plain_sse = (activations @ (weight - plain_weight).T).square().sum()
    feedback_sse = (
        activations @ (weight - feedback_weight).T
    ).square().sum()

    # With deterministic fit/holdout split, LDLQ trains on half rows, so improvement is slightly less than monolithic full-data LDLQ; relax threshold accordingly
    assert float(feedback_sse / plain_sse) < 0.6


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
@pytest.mark.parametrize("format_name", ["NVFP4_CB_K12", "FP8_CB_K28"])
@pytest.mark.parametrize("strategy", ["chunked", "threaded"])
def test_batched_expert_ldlq_is_bit_identical_to_serial(
    device, format_name, strategy, monkeypatch
):
    """The production expert batch may never change a per-unit encoding."""
    # Force multiple expert chunks so this pins both vectorization and the
    # chunk concatenation used by large packed-MoE stacks.
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_EXPERT_BATCH", "3")
    monkeypatch.setenv(
        "PRISMAQUANT_CB_LDLQ_FEEDER_THREADS",
        "4" if strategy == "threaded" else "0",
    )
    monkeypatch.setenv(
        "PRISMAQUANT_CB_LDLQ_BATCH_STREAMS",
        "2" if strategy == "chunked" else "1",
    )
    generator = torch.Generator(device="cpu").manual_seed(19)
    experts, rows, columns = 8, 8, 256
    weight = (
        torch.randn(experts, rows, columns, generator=generator) * 0.08
    ).to(device=device, dtype=torch.bfloat16)
    col_weights = (
        torch.rand(experts, 1, columns, generator=generator) + 0.05
    ).to(device)
    activation_rows = tuple(
        torch.randn(9 + expert % 4, columns, generator=generator).to(device)
        for expert in range(experts)
    )
    spec = fr.get_format(format_name)
    grid = "fp4" if format_name.startswith("NVFP4") else "fp8"
    k = int(format_name.rsplit("K", 1)[1])
    fields = cb_fields_for_context(
        spec,
        weight,
        context=CBSerializationContext.production(encode_tier="fast"),
        col_weights=col_weights,
    )

    serial = cb.ldlq_reassign_cb_fields(
        weight,
        fields,
        col_weights,
        activation_rows,
        grid=grid,
        mode="product",
        batch_experts=False,
    )
    batched = cb.ldlq_reassign_cb_fields(
        weight,
        fields,
        col_weights,
        activation_rows,
        grid=grid,
        mode="product",
        batch_experts=True,
    )

    for key in ("indices", "scales", "scale_super", "scale_sub"):
        if key in serial:
            assert torch.equal(serial[key], batched[key]), (
                f"{format_name}/{device}: {key} differs"
            )
    serial_reconstruction = cb.nvfp4_cb_reconstruct(
        serial, k, grid=grid, mode="product"
    )
    batched_reconstruction = cb.nvfp4_cb_reconstruct(
        batched, k, grid=grid, mode="product"
    )
    assert torch.equal(serial_reconstruction, batched_reconstruction)
    serial_mse = torch.stack([
        (
            activation_rows[expert]
            @ (weight[expert].float() - serial_reconstruction[expert]).T
        ).square().mean()
        for expert in range(experts)
    ])
    batched_mse = torch.stack([
        (
            activation_rows[expert]
            @ (weight[expert].float() - batched_reconstruction[expert]).T
        ).square().mean()
        for expert in range(experts)
    ])
    assert torch.equal(serial_mse, batched_mse)
