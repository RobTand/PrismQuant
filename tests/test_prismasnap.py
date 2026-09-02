from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import prismaquant.export_native_compressed as native
import prismaquant.prismasnap as snap
import prismaquant.prismasnap_checkpoint as checkpoint
from prismaquant.prismasnap_contract import refuse_prismasnap_for_unvalidated_lane


def _search_consumers() -> tuple[list[snap.PrismaSnapConsumer], torch.Tensor]:
    generator = torch.Generator().manual_seed(20260825)
    importance = torch.logspace(-2, 2, 32)
    shared_importance = torch.linspace(0.25, 2.0, 32)
    consumers = [
        snap.PrismaSnapConsumer(
            name="renamed.first",
            weight=torch.randn(32, 32, generator=generator) * 0.19,
            importance=importance,
            mode="column_inverse",
        ),
        snap.PrismaSnapConsumer(
            name="renamed.second",
            weight=torch.randn(32, 32, generator=generator) * 1.7,
            importance=shared_importance,
            mode="column_inverse",
        ),
    ]
    return consumers, importance


def test_explicit_codec_is_environment_independent_and_default_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(7, 32, generator=generator) * 0.31
    monkeypatch.delenv(native.NVFP4_SCALE_RULE_ENV, raising=False)
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING", raising=False)
    monkeypatch.setattr(native, "_NVFP4_SCALE_RULE", None)

    legacy = native._rtn_dequant_nvfp4(weight, group_size=16)
    explicit_before = native.render_nvfp4_dequant(weight)
    grouped = weight.reshape(7, 2, 16)
    _legacy_scale, legacy_global = native._select_nvfp4_pack_scales_and_global(
        grouped
    )
    explicit_global_before = native.nvfp4_global_real(weight)
    explicit_joint_before = native.render_nvfp4_dequant(
        weight,
        scale_rule=native.NVFP4_SCALE_RULE_JOINT_MSE,
        joint_scale_levels=(6.0, 4.0),
    )
    torch.testing.assert_close(explicit_before, legacy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        explicit_global_before, legacy_global, rtol=0.0, atol=0.0
    )

    # Both historical process-global selectors now disagree with the explicit
    # request.  The receipt-selected codec must nevertheless be byte-stable.
    monkeypatch.setenv(native.NVFP4_SCALE_RULE_ENV, "joint_mse")
    monkeypatch.setenv("PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING", "1")
    monkeypatch.setattr(native, "_NVFP4_SCALE_RULE", "four_over_six_mse")
    monkeypatch.setattr(native, "_NVFP4_JOINT_SCALE_LEVELS", (6.0, 3.0, 1.0))
    explicit_after = native.render_nvfp4_dequant(
        weight,
        scale_rule=native.NVFP4_SCALE_RULE_STATIC_6,
        snapped_scale_scoring=False,
    )
    explicit_global_after = native.nvfp4_global_real(
        weight,
        scale_rule=native.NVFP4_SCALE_RULE_STATIC_6,
        snapped_scale_scoring=False,
    )
    explicit_joint_after = native.render_nvfp4_dequant(
        weight,
        scale_rule=native.NVFP4_SCALE_RULE_JOINT_MSE,
        snapped_scale_scoring=False,
        joint_scale_levels=(6.0, 4.0),
    )
    torch.testing.assert_close(explicit_after, explicit_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        explicit_global_after, explicit_global_before, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        explicit_joint_after, explicit_joint_before, rtol=0.0, atol=0.0
    )


def test_search_is_deterministic_and_no_op_is_a_hard_upper_bound() -> None:
    consumers, importance = _search_consumers()
    config = snap.PrismaSnapSearchConfig(
        alphas=(0.0, 0.25, 0.5),
        max_rounds=3,
        stage=True,
        polish=True,
        polish_top=2,
        polish_pool=2,
        scale_rule="static_6",
    )

    scale_a, stats_a = snap.search_diagonal_scale(
        consumers, importance, config=config
    )
    scale_b, stats_b = snap.search_diagonal_scale(
        consumers, importance, config=config
    )

    assert torch.equal(scale_a, scale_b)
    assert stats_a == stats_b
    assert stats_a["error_final"] <= stats_a["error_baseline"]
    identity_error = snap.measured_render_objective(
        consumers, torch.ones_like(scale_a), config
    )
    assert stats_a["error_baseline"] == identity_error
    assert stats_a["candidate_count"] == len(config.alphas)


def test_measured_v1_derives_one_global_per_logical_tensor() -> None:
    low = snap.PrismaSnapConsumer(
        name="fused.low",
        weight=torch.full((16, 16), 0.03125),
        importance=torch.ones(16),
        mode="row",
    )
    high_stationary = snap.PrismaSnapConsumer(
        name="fused.high_stationary",
        weight=torch.linspace(-9.0, 9.0, 256).reshape(16, 16),
        importance=torch.ones(16),
        mode="stationary",
    )
    independent = snap.PrismaSnapConsumer(
        name="independent",
        weight=torch.full((16, 16), 0.5),
        importance=torch.ones(16),
        mode="row",
    )
    config = snap.PrismaSnapSearchConfig(
        alphas=(0.0,),
        max_rounds=1,
        stage=False,
        polish=False,
        polish_top=0,
        polish_pool=0,
        scale_rule="static_6",
    )
    scale = torch.ones(16, dtype=torch.float64)

    globals_by_group = snap._shared_globals(
        [low, high_stationary, independent], scale, config
    )
    torch.testing.assert_close(
        globals_by_group["fused.low"],
        native.nvfp4_global_real(low.weight),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        globals_by_group["fused.high_stationary"],
        native.nvfp4_global_real(high_stationary.weight),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        globals_by_group["independent"],
        native.nvfp4_global_real(independent.weight),
        rtol=0.0,
        atol=0.0,
    )


def test_fp64_dense_folds_are_exact_for_norm_and_up_down_seams() -> None:
    generator = torch.Generator().manual_seed(99)
    hidden = 16
    intermediate = 32
    tokens = torch.randint(-4, 5, (5, hidden), generator=generator).to(torch.float64)
    norm = torch.randint(1, 5, (hidden,), generator=generator).to(torch.float64)
    projection = torch.randint(
        -5, 6, (hidden, hidden), generator=generator
    ).to(torch.float64)
    norm_scale = torch.tensor([0.5, 1.0, 2.0, 4.0] * 4, dtype=torch.float64)

    folded_norm = snap.apply_diagonal_transform(norm, norm_scale, "multiply", 0)
    folded_projection = snap.apply_diagonal_transform(
        projection, norm_scale, "divide", 1
    )
    before_norm = (tokens * norm) @ projection.T
    after_norm = (tokens * folded_norm) @ folded_projection.T
    assert folded_norm.dtype == torch.float64
    assert folded_projection.dtype == torch.float64
    assert torch.equal(after_norm, before_norm)

    # Qwen3.5/Qwen3.8 stores p = gamma - 1, so the exact checkpoint
    # transform is p' = (p + 1) * d - 1, not p' = p * d.
    offset_parameter = norm - 1.0
    folded_offset_parameter = snap.apply_diagonal_transform(
        offset_parameter,
        norm_scale,
        "affine_multiply",
        0,
        parameter_offset=1.0,
    )
    before_offset = (tokens * (offset_parameter + 1.0)) @ projection.T
    after_offset = (
        tokens * (folded_offset_parameter + 1.0)
    ) @ folded_projection.T
    assert torch.equal(after_offset, before_offset)

    gate = torch.randint(
        -4, 5, (intermediate, hidden), generator=generator
    ).to(torch.float64)
    up = torch.randint(
        -4, 5, (intermediate, hidden), generator=generator
    ).to(torch.float64)
    down = torch.randint(
        -4, 5, (hidden, intermediate), generator=generator
    ).to(torch.float64)
    updown_scale = torch.tensor(
        [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 0.125, 16.0] * 4,
        dtype=torch.float64,
    )
    folded_up = snap.apply_diagonal_transform(up, updown_scale, "multiply", 0)
    folded_down = snap.apply_diagonal_transform(
        down, updown_scale, "divide", 1
    )
    gated = F.silu(tokens @ gate.T)
    before_mlp = (gated * (tokens @ up.T)) @ down.T
    after_mlp = (gated * (tokens @ folded_up.T)) @ folded_down.T
    assert torch.equal(after_mlp, before_mlp)


def test_multiple_tensor_folds_reproduce_sequential_bf16_v1_rounding() -> None:
    generator = torch.Generator().manual_seed(20260825)
    tensor = torch.randn(32, 16, generator=generator).to(torch.bfloat16)
    row_scale = torch.linspace(0.55, 1.75, 32, dtype=torch.float64)
    col_scale = torch.linspace(0.6, 1.4, 16, dtype=torch.float64)

    work = snap.apply_diagonal_transform(
        tensor, col_scale, "divide", 1, output_dtype=torch.bfloat16
    )
    work = snap.apply_diagonal_transform(
        work, row_scale, "multiply", 0, output_dtype=torch.bfloat16
    )
    materialized = work
    expected = (
        (tensor.to(torch.float64) / col_scale.view(1, -1))
        .to(torch.bfloat16)
        .to(torch.float64)
        * row_scale.view(-1, 1)
    ).to(torch.bfloat16)
    assert torch.equal(materialized, expected)


def test_transform_refuses_finite_fp64_value_that_overflows_checkpoint_dtype() -> None:
    tensor = torch.tensor([3.0e38], dtype=torch.float32)
    scale = torch.tensor([2.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="overflowed"):
        snap.apply_diagonal_transform(
            tensor, scale, "multiply", 0, output_dtype=torch.bfloat16
        )


def test_non_native_lanes_refuse_any_prismasnap_source_marker(tmp_path) -> None:
    (tmp_path / "prismasnap_provenance.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not admitted.*Gridbook"):
        refuse_prismasnap_for_unvalidated_lane(tmp_path, lane="Gridbook/codebook")


# `test_programmatic_codebook_exporters_gate_before_opening_output` was deleted
# on 2026-09-02. It proved that `export_nvfp4_cb` and
# `export_nvfp4_cb_streaming` call `refuse_prismasnap_for_unvalidated_lane`
# BEFORE creating their output directory, so a refused export leaves no
# half-written artifact. Both exporters are in
# archive/gridbook_lane_2026-09-02/ and no live exporter reaches the codebook
# lane. The refusal function itself is still pinned directly, one test above.


class _RenamedProfile:
    name = "renamed_dense"

    @staticmethod
    def fused_sibling_group(qname: str) -> str | None:
        if qname.endswith((".branch_a", ".branch_b")):
            return "model.layers.0.arbitrary.joined_pair"
        return None

    @staticmethod
    def fused_sibling_leaf_mapping() -> dict[str, tuple[str, ...]]:
        return {"joined_pair": ("branch_a", "branch_b")}

    @staticmethod
    def source_tensor_name(qname: str) -> str:
        return qname

    @staticmethod
    def rms_norm_parameter_offset() -> float:
        return 0.0


class _MetadataCheckpoint:
    def __init__(
        self,
        metadata: dict[str, tuple[tuple[int, ...], str]],
        *,
        scaled: frozenset[str] = frozenset(),
    ) -> None:
        self._metadata = metadata
        # Which weights the index pairs with a block ``weight_scale_inv``.
        # Membership is the index's, never inferred from a dtype: an FP8
        # weight with no paired scale is not block-dequantable and must be
        # refused rather than silently read as if it carried one.
        self._scaled = scaled

    def metadata(self, key: str) -> tuple[tuple[int, ...], str]:
        return self._metadata[key]

    def fp8_scale_key(self, key: str) -> str | None:
        return f"{key}_scale_inv" if key in self._scaled else None


def _renamed_graph_fixture(
    *, weight_dtype: str = "BF16", scale_paired: bool = True
) -> tuple[dict[str, dict[str, object]], _MetadataCheckpoint]:
    hidden = 16
    intermediate = 32
    input_importance = np.linspace(0.5, 1.5, hidden, dtype=np.float32)
    post_importance = np.linspace(1.5, 0.5, hidden, dtype=np.float32)
    stats = {
        "model.layers.0.arbitrary.reader_x": {
            "act_sq_sum": input_importance.copy(),
            "in_features": hidden,
            "out_features": hidden,
        },
        "model.layers.0.arbitrary.reader_y": {
            "act_sq_sum": input_importance.copy(),
            "in_features": hidden,
            "out_features": hidden,
        },
        "model.layers.0.arbitrary.branch_a": {
            "act_sq_sum": post_importance.copy(),
            "in_features": hidden,
            "out_features": intermediate,
        },
        "model.layers.0.arbitrary.branch_b": {
            "act_sq_sum": post_importance.copy(),
            "in_features": hidden,
            "out_features": intermediate,
        },
        "model.layers.0.arbitrary.return_path": {
            "act_sq_sum": np.ones(intermediate, dtype=np.float32),
            "in_features": intermediate,
            "out_features": hidden,
        },
    }
    metadata: dict[str, tuple[tuple[int, ...], str]] = {
        "model.layers.0.input_layernorm.weight": ((hidden,), "BF16"),
        "model.layers.0.post_attention_layernorm.weight": ((hidden,), "BF16"),
    }
    for qname, row in stats.items():
        metadata[f"{qname}.weight"] = (
            (int(row["out_features"]), int(row["in_features"])),
            weight_dtype if qname.endswith("reader_x") else "BF16",
        )
    scaled = frozenset(
        key
        for key, (_shape, dtype) in metadata.items()
        if dtype == "F8_E4M3" and scale_paired
    )
    return stats, _MetadataCheckpoint(metadata, scaled=scaled)


def test_dense_graph_discovery_uses_activation_equivalence_not_name_allowlist() -> None:
    stats, source = _renamed_graph_fixture()
    graph = checkpoint._discover_dense_layer_graph(
        0,
        hidden_size=16,
        stats=stats,
        profile=_RenamedProfile(),
        checkpoint=source,  # type: ignore[arg-type]
    )

    assert graph["input_consumers"] == [
        "model.layers.0.arbitrary.reader_x",
        "model.layers.0.arbitrary.reader_y",
    ]
    assert graph["post_consumers"] == [
        "model.layers.0.arbitrary.branch_a",
        "model.layers.0.arbitrary.branch_b",
    ]
    assert graph["gate"].endswith("branch_a")
    assert graph["up"].endswith("branch_b")
    assert graph["down"].endswith("return_path")


def _discover(source, stats):
    return checkpoint._discover_dense_layer_graph(
        0,
        hidden_size=16,
        stats=stats,
        profile=_RenamedProfile(),
        checkpoint=source,  # type: ignore[arg-type]
    )


def test_dense_graph_discovery_accepts_a_block_scaled_fp8_source() -> None:
    """A native-FP8 source is planned, not refused.

    PrismaSnap folds a per-CHANNEL diagonal; an FP8 checkpoint carries a
    per-BLOCK scale, so the fold cannot be absorbed into the scale and its
    result is BF16 -- which is what the fold-fidelity gate serves anyway.
    The plan therefore covers the same seams it would on a BF16 source.
    """
    stats, source = _renamed_graph_fixture(weight_dtype="F8_E4M3")
    graph = _discover(source, stats)
    assert graph["source_weights"]["model.layers.0.arbitrary.reader_x"] == (
        "model.layers.0.arbitrary.reader_x.weight"
    )


def test_dense_graph_discovery_refuses_fp8_without_a_paired_block_scale() -> None:
    """Per-tensor-scale FP8 is not block-dequantable, so it must fail closed.

    The failure is keyed on the index pairing, not on the dtype: reading an
    unpaired FP8 tensor as if it carried a block scale is silent corruption.
    """
    stats, source = _renamed_graph_fixture(
        weight_dtype="F8_E4M3", scale_paired=False
    )
    with pytest.raises(RuntimeError, match="no weight_scale_inv"):
        _discover(source, stats)


def test_dense_graph_discovery_refuses_an_unliftable_source_dtype() -> None:
    """Widening to FP8 must not widen to everything.

    A source PrismaSnap cannot lift exactly into BF16 is a source whose fold
    it could not attest, so the census gate still refuses it by name.
    """
    stats, source = _renamed_graph_fixture(weight_dtype="I8")
    with pytest.raises(RuntimeError, match="cannot lift source dtype"):
        _discover(source, stats)
