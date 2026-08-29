from __future__ import annotations

import json
import re

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant.cb_export_config import build_cb_scheme, build_quant_config
from prismaquant.nvfp4_activation_contract import (
    CALIBRATION_SOURCE_PACKED_EXPERT_RENDER,
    CALIBRATION_SOURCE_PARENT_MODULE_CACHE,
    CALIBRATION_SOURCE_SUPPLEMENTAL_MAX_ABS,
    CALIBRATION_SOURCE_SUPPLEMENTAL_MODULE_INPUT,
    CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
    CALIBRATION_SOURCE_TARGET_CACHE,
    FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
    LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
    NVFP4_ACTIVATION_CONTRACT_KEY,
    NVFP4_ACTIVATION_CONTRACT_SCHEMA,
    NVFP4_ACTIVATION_CONTRACT_SCHEMA_V2,
    NVFP4_ACTIVATION_EXECUTION,
    NVFP4_INPUT_GLOBAL_SCALE_SUFFIX,
    NVFP4_ROUTED_MOE_STAGE_KEY,
    NVFP4_ROUTED_MOE_STAGE_SCHEMA,
    UNCALIBRATED_INPUT_GLOBAL_SCALE,
    build_execution_contract,
    build_routed_moe_stage_attestation,
    calibrated_input_global_scales,
    calibrated_input_global_scales_with_sources,
    fused_dense_group,
    fused_sibling_group_key,
    group_fused_sibling_targets,
    input_global_scale_from_max_abs,
    nvfp4_activation_qdq_served,
    resolve_input_global_scale_value,
    routed_moe_stage,
    select_mse_grid_input_global_scale,
    stage_values_sha256,
    target_values_sha256,
    unify_fused_sibling_input_global_scales,
    unify_fused_sibling_max_abs,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_serialization_context_stamp,
    cb_tensor_payload_breakdown,
    cb_tensor_serialization_stamp,
)


class _FusedProfile:
    @staticmethod
    def fused_sibling_group(name):
        if name in {"layer.q_proj", "layer.k_proj", "layer.v_proj"}:
            return "layer.qkv"
        return None


def _write_activation(cache_dir, name, inputs, row_indices=None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
    blob = {"name": name, "inputs": inputs}
    if row_indices is not None:
        blob["row_indices"] = row_indices
    torch.save(blob, cache_dir / filename)


def test_formula_policies_are_explicit_and_f32_rounded():
    legacy = input_global_scale_from_max_abs(
        3.0,
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    )
    full = input_global_scale_from_max_abs(
        3.0,
        policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
    )
    assert legacy == 2.0
    assert full == 896.0
    with pytest.raises(ValueError, match="activation samples"):
        input_global_scale_from_max_abs(
            3.0,
            policy=MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
        )
    with pytest.raises(ValueError, match="finite and > 0"):
        input_global_scale_from_max_abs(
            0.0,
            policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        )


def test_uncalibrated_fallback_is_explicit_and_legacy_only():
    assert input_global_scale_from_max_abs(
        0.0,
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        nonpositive_fallback=UNCALIBRATED_INPUT_GLOBAL_SCALE,
    ) == 1.0
    with pytest.raises(ValueError, match="finite and > 0"):
        input_global_scale_from_max_abs(
            float("nan"),
            policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
            nonpositive_fallback=UNCALIBRATED_INPUT_GLOBAL_SCALE,
        )

    with pytest.raises(ValueError, match="no calibrated"):
        resolve_input_global_scale_value(target="layer.q_proj")
    assert resolve_input_global_scale_value(
        target="layer.q_proj",
        allow_uncalibrated_fallback=True,
    ) == UNCALIBRATED_INPUT_GLOBAL_SCALE
    assert resolve_input_global_scale_value(
        3.0,
        target="layer.q_proj",
        calibrated_scales={"layer.q_proj": 2.0},
        allow_uncalibrated_fallback=True,
    ) == 3.0
    assert resolve_input_global_scale_value(
        target="layer.q_proj",
        calibrated_scales={"layer.q_proj": 2.0},
        allow_uncalibrated_fallback=True,
    ) == 2.0


def test_one_grouping_primitive_drives_max_and_reciprocal_joins():
    q = "model.layers.2.self_attn.q_proj"
    k = "model.layers.2.self_attn.k_proj"
    down = "model.layers.2.mlp.down_proj"

    fallback = fused_dense_group(q)
    assert fallback == (
        "model.layers.2",
        ("q_proj", "k_proj", "v_proj"),
    )
    assert fused_sibling_group_key(q) == fused_sibling_group_key(k)
    groups = group_fused_sibling_targets([q, k, down])
    assert sorted(map(tuple, groups.values())) == sorted([(q, k), (down,)])

    max_abs = unify_fused_sibling_max_abs({q: 1.0, k: 4.0, down: 3.0})
    scales = unify_fused_sibling_input_global_scales(
        {q: 0.5, k: 0.25, down: 0.75}
    )
    assert max_abs == {q: 4.0, k: 4.0, down: 3.0}
    assert scales == {q: 0.25, k: 0.25, down: 0.75}


def test_grouping_profile_errors_are_strict_unless_legacy_opts_in():
    class BrokenProfile:
        @staticmethod
        def fused_sibling_group(_name):
            raise RuntimeError("profile unavailable")

    name = "model.layers.0.self_attn.q_proj"
    with pytest.raises(RuntimeError, match="profile unavailable"):
        fused_sibling_group_key(name, profile=BrokenProfile())
    assert fused_sibling_group_key(
        name,
        profile=BrokenProfile(),
        tolerate_profile_errors=True,
    ) == "model.layers.0::__fused__:q_proj,k_proj,v_proj"


def test_profile_leaf_mapping_and_direct_group_share_key_api():
    class LeafProfile:
        @staticmethod
        def fused_sibling_group(_name):
            return None

        @staticmethod
        def fused_sibling_leaf_mapping():
            return {"ab_proj": ("a_proj", "b_proj")}

    profile = LeafProfile()
    assert fused_sibling_group_key(
        "layer.a_proj", profile=profile
    ) == fused_sibling_group_key("layer.b_proj", profile=profile)


def test_e2m1_midpoints_use_encoded_index_even_rne():
    # Include 6 so the stored group scale is exactly 1 at G=1.
    values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
    x = torch.tensor(values + [-value for value in values]).reshape(1, 16)
    expected_positive = torch.tensor(
        [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]
    )
    expected = torch.cat((expected_positive, -expected_positive)).reshape(1, 16)
    torch.testing.assert_close(
        nvfp4_activation_qdq_served(x, 1.0),
        expected,
        rtol=0,
        atol=0,
    )


def test_ue4m3_scale_underflow_has_no_minimum_clamp():
    midpoint_to_first_subnormal = 6.0 * (2.0 ** -10)
    at_tie = torch.full((1, 16), midpoint_to_first_subnormal)
    just_above = torch.full(
        (1, 16),
        torch.nextafter(
            torch.tensor(midpoint_to_first_subnormal),
            torch.tensor(float("inf")),
        ).item(),
    )
    assert torch.count_nonzero(nvfp4_activation_qdq_served(at_tie, 1.0)) == 0
    assert torch.count_nonzero(
        nvfp4_activation_qdq_served(just_above, 1.0)
    ) == 16


def test_static_qdq_is_chunk_independent():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(17, 32, generator=generator) * 0.7
    whole = nvfp4_activation_qdq_served(x, 128.0)
    chunked = torch.cat([
        nvfp4_activation_qdq_served(x[:3], 128.0),
        nvfp4_activation_qdq_served(x[3:11], 128.0),
        nvfp4_activation_qdq_served(x[11:], 128.0),
    ])
    assert torch.equal(whole, chunked)


def test_mse_grid_contains_both_formula_endpoints():
    generator = torch.Generator().manual_seed(11)
    sample = torch.randn(23, 32, generator=generator)
    max_abs = float(sample.abs().max())
    legacy = input_global_scale_from_max_abs(
        max_abs, policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY
    )
    full = input_global_scale_from_max_abs(
        max_abs, policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
    )
    selected = select_mse_grid_input_global_scale([sample])

    def mse(scale):
        return float((nvfp4_activation_qdq_served(sample, scale) - sample)
                     .square().mean())

    assert mse(selected) <= min(mse(legacy), mse(full)) + 1e-12


@pytest.mark.parametrize(
    "policy",
    [FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY, MSE_GRID_INPUT_GLOBAL_SCALE_POLICY],
)
def test_fused_siblings_fit_union_and_emit_identical_f32(tmp_path, policy):
    cache = tmp_path / "act"
    q = torch.linspace(-1.0, 1.0, 64).reshape(2, 32)
    k = torch.linspace(-4.0, 4.0, 96).reshape(3, 32)
    # Deliberately different reservoirs/row identities: runtime still merges.
    _write_activation(cache, "layer.q_proj", q, torch.tensor([1, 9]))
    _write_activation(cache, "layer.k_proj", k, torch.tensor([2, 3, 7]))
    scales = calibrated_input_global_scales(
        ["layer.q_proj", "layer.k_proj"],
        activation_cache_dir=cache,
        policy=policy,
        profile=_FusedProfile(),
    )
    assert scales["layer.q_proj"] == scales["layer.k_proj"]
    if policy == FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY:
        assert scales["layer.q_proj"] == input_global_scale_from_max_abs(
            4.0,
            policy=policy,
        )
    else:
        assert scales["layer.q_proj"] == select_mse_grid_input_global_scale(
            [q, k]
        )


def test_missing_calibration_fails_closed(tmp_path):
    cache = tmp_path / "act"
    _write_activation(cache, "layer.q_proj", torch.ones(2, 32))
    with pytest.raises(ValueError, match="no calibrated input"):
        calibrated_input_global_scales(
            ["layer.q_proj", "layer.k_proj"],
            activation_cache_dir=cache,
            policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
            profile=_FusedProfile(),
        )


def test_contract_digest_framing_has_pinned_vector():
    assert target_values_sha256(
        {"a": 1.0, "b": 2.0},
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    ) == "5207c30737409ae6d16586f1f169efc8f56948bee51031e1610683f0fee08d0f"


def test_contract_uses_the_canonical_tensor_suffix_api():
    record, _ = build_execution_contract(
        {"layer.q_proj": 2.0},
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    )
    assert record["tensor_suffix"] == NVFP4_INPUT_GLOBAL_SCALE_SUFFIX


def test_accounting_is_keyed_to_static_contract_variant():
    static = CBSerializationContext.production()
    old = CBSerializationContext(
        scale_coding="two_tier",
        codebook_source="lattice",
    )
    fp4_static = cb_tensor_payload_breakdown(
        "NVFP4_CB_K16", (8, 256), qname="w", context=static
    )
    fp4_old = cb_tensor_payload_breakdown(
        "NVFP4_CB_K16", (8, 256), qname="w", context=old
    )
    fp8_static = cb_tensor_payload_breakdown(
        "FP8_CB_K36", (8, 256), qname="w", context=static
    )
    assert fp4_static["input_global_scale_bytes"] == 4
    assert fp4_static["tensor_payload_bytes"] == (
        fp4_old["tensor_payload_bytes"] + 4
    )
    assert fp8_static["input_global_scale_bytes"] == 0


def test_config_has_one_top_level_contract_and_fp4_only_reference():
    codebooks = {
        ("lattice", "NVFP4_CB_K12"): (
            torch.zeros(64, 4), torch.zeros(64, 4)
        ),
        ("lattice", "FP8_CB_K40"): tuple(
            torch.zeros(1024, 2) for _ in range(4)
        ),
    }
    record, _ = build_execution_contract(
        {"fp4": 8.0},
        policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
    )
    config = build_quant_config(
        assignment={"fp4": "NVFP4_CB_K12", "fp8": "FP8_CB_K40"},
        cb_targets={
            "fp4": ("fp4", "product", 12),
            "fp8": ("fp8", "product", 40),
        },
        source_targets=[],
        stock_targets={},
        by_group={
            ("lattice", "NVFP4_CB_K12"): ["fp4"],
            ("lattice", "FP8_CB_K40"): ["fp8"],
        },
        codebooks=codebooks,
        col_weights={},
        codebook_tensors_by_name={},
        ignore=[],
        codebook_file=None,
        scale_coding="two_tier",
        codebook_source="lattice",
        serialized_payload_summary={"total_bytes": 0},
        serialization_context=CBSerializationContext.production(),
        cb_render_identity=None,
        activation_execution_contract=record,
        git_commit="test",
    )
    assert config["execution_contracts"] == {
        NVFP4_ACTIVATION_CONTRACT_KEY: record
    }
    assert "nvfp4_activation_contract" not in config["provenance"]
    schemes = {
        group["scheme"]["grid"]: group["scheme"]
        for group in config["config_groups"].values()
    }
    assert schemes["fp4"]["activation_contract"] == (
        NVFP4_ACTIVATION_CONTRACT_KEY
    )
    assert "activation_contract" not in schemes["fp8"]


def _production_layer_config(
    path,
    qname,
    weight,
    col_weights,
    *,
    extra_assignment=None,
    context=None,
):
    from prismaquant.production_weight_cache import (
        bind_cb_render_identity_source_weights,
        build_production_cache_cb_render_identity,
    )

    context = context or CBSerializationContext.production()
    identity = build_production_cache_cb_render_identity(
        {qname: "NVFP4_CB_K16"},
        cb_serialization_context=context,
        col_weights={qname: col_weights},
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    identity = bind_cb_render_identity_source_weights(
        identity,
        {qname: weight},
    )
    stamp = cb_serialization_context_stamp(
        context,
        formats=["NVFP4_CB_K16"],
    )
    payload = {
        qname: {
            "data_type": "nvfp4_cb",
            "cb_k": 16,
            "cb_serialized_identity": cb_tensor_serialization_stamp(
                "NVFP4_CB_K16",
                tuple(weight.shape),
                qname=qname,
                context=context,
            ),
        },
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "cb_serialized_payload": stamp,
            "cb_render_identity": identity,
        },
    }
    payload.update(extra_assignment or {})
    path.write_text(json.dumps(payload))


def test_resident_and_streaming_export_same_static_scalar_and_contract(tmp_path):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb
    from prismaquant.export_nvfp4_cb_streaming import export_nvfp4_cb_streaming

    source = tmp_path / "source"
    source.mkdir()
    qname = "model.layers.0.self_attn.o_proj"
    stock_name = "model.layers.0.mlp.down_proj"
    generator = torch.Generator().manual_seed(123)
    weight = (torch.randn(8, 256, generator=generator) * 0.2).to(
        torch.bfloat16
    )
    stock_weight = (
        torch.randn(8, 256, generator=generator) * 0.2
    ).to(torch.bfloat16)
    save_file(
        {
            qname + ".weight": weight,
            stock_name + ".weight": stock_weight,
        },
        str(source / "model.safetensors"),
    )
    (source / "config.json").write_text(json.dumps({
        "architectures": ["ContractTiny"],
        "hidden_size": 256,
    }))
    col_weights = torch.linspace(0.5, 1.5, 256)
    assignment = tmp_path / "assignment.json"
    _production_layer_config(
        assignment,
        qname,
        weight,
        col_weights,
        extra_assignment={stock_name: "NVFP4"},
    )
    activation_cache = tmp_path / "act"
    _write_activation(
        activation_cache,
        qname,
        torch.randn(13, 256, generator=generator) * 0.4,
    )
    _write_activation(
        activation_cache,
        stock_name,
        torch.randn(11, 256, generator=generator) * 0.6,
    )

    resident = tmp_path / "resident"
    streaming = tmp_path / "streaming"
    common = dict(
        activation_cache_dir=activation_cache,
        activation_scale_policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        device="cpu",
    )
    export_nvfp4_cb(
        source,
        assignment,
        resident,
        {qname: col_weights},
        **common,
    )
    export_nvfp4_cb_streaming(
        source,
        assignment,
        streaming,
        {qname: col_weights},
        **common,
    )

    resident_tensors = load_file(str(resident / "model.safetensors"))
    streaming_tensors = load_file(str(streaming / "model.safetensors"))
    for target in (qname, stock_name):
        scalar_name = target + ".input_global_scale"
        assert resident_tensors[scalar_name].dtype == torch.float32
        assert tuple(resident_tensors[scalar_name].shape) == (1,)
        assert torch.equal(
            resident_tensors[scalar_name],
            streaming_tensors[scalar_name],
        )
    resident_config = json.loads((resident / "quant_config.json").read_text())
    streaming_config = json.loads((streaming / "quant_config.json").read_text())
    assert resident_config["execution_contracts"] == (
        streaming_config["execution_contracts"]
    )
    record = resident_config["execution_contracts"][
        NVFP4_ACTIVATION_CONTRACT_KEY
    ]
    assert record["schema"] == NVFP4_ACTIVATION_CONTRACT_SCHEMA
    assert record["contract"] == NVFP4_ACTIVATION_EXECUTION
    assert record["target_count"] == 2
    assert record["target_names"] == sorted((qname, stock_name))
    emitted_scalar_targets = sorted(
        name.removesuffix(".input_global_scale")
        for name in resident_tensors
        if name.endswith(".input_global_scale")
    )
    assert emitted_scalar_targets == record["target_names"]
    assert resident_config["provenance"]["serialized_payload"][
        "input_global_scale_bytes"
    ] == 4


def test_research_export_omits_static_contract_and_scalar(tmp_path):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    source = tmp_path / "source"
    source.mkdir()
    qname = "model.layers.0.self_attn.o_proj"
    weight = torch.randn(4, 256).to(torch.bfloat16)
    save_file({qname + ".weight": weight}, str(source / "model.safetensors"))
    (source / "config.json").write_text("{}")
    assignment = tmp_path / "assignment.json"
    assignment.write_text(json.dumps({qname: "NVFP4_CB_K16"}))
    out = tmp_path / "out"
    export_nvfp4_cb(
        source,
        assignment,
        out,
        {qname: torch.ones(256)},
        device="cpu",
        allow_unstamped_research=True,
    )
    config = json.loads((out / "quant_config.json").read_text())
    tensors = load_file(str(out / "model.safetensors"))
    assert "execution_contracts" not in config
    assert qname + ".input_global_scale" not in tensors
    group = next(iter(config["config_groups"].values()))
    assert "activation_contract" not in group["scheme"]


def test_v2_stamped_export_remains_fused_ineligible(tmp_path):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    source = tmp_path / "source"
    source.mkdir()
    qname = "model.layers.0.self_attn.o_proj"
    weight = torch.randn(4, 256).to(torch.bfloat16)
    save_file({qname + ".weight": weight}, str(source / "model.safetensors"))
    (source / "config.json").write_text("{}")
    assignment = tmp_path / "assignment.json"
    old_context = CBSerializationContext(
        scale_coding="two_tier",
        codebook_source="lattice",
    )
    _production_layer_config(
        assignment,
        qname,
        weight,
        torch.ones(256),
        context=old_context,
    )
    out = tmp_path / "out"
    export_nvfp4_cb(
        source,
        assignment,
        out,
        {qname: torch.ones(256)},
        device="cpu",
    )
    config = json.loads((out / "quant_config.json").read_text())
    tensors = load_file(str(out / "model.safetensors"))
    assert config["provenance"]["serialized_payload"]["schema"].endswith(
        ".v2"
    )
    assert "execution_contracts" not in config
    assert qname + ".input_global_scale" not in tensors
    group = next(iter(config["config_groups"].values()))
    assert "activation_contract" not in group["scheme"]


# ---------------------------------------------------------------------------
# ROADMAP K0.2 — routed-MoE stage attestation
# ---------------------------------------------------------------------------

_W13 = "model.layers.0.mlp.experts.gate_up_proj"
_W2 = "model.layers.0.mlp.experts.down_proj"
_STAGE_SOURCES = {
    _W13: CALIBRATION_SOURCE_PARENT_MODULE_CACHE,
    _W2: CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
}


class _PackedExpertProfile:
    """Profile whose on-disk packed-expert leaves are LFM2.5's w1/w3/w2."""

    @staticmethod
    def packed_expert_role_group(qname):
        leaf = str(qname).rsplit(".", 1)[-1]
        if leaf in {"w1", "w3"}:
            return "gate_up_proj"
        if leaf == "w2":
            return "down_proj"
        return None

    @staticmethod
    def source_tensor_name(name):
        return name


def _stage_section(scales=None, *, sources=None, policy=None, profile=None):
    scales = {_W13: 1.0, _W2: 2.0} if scales is None else scales
    return build_routed_moe_stage_attestation(
        scales,
        policy=policy or LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        calibration_sources=sources or _STAGE_SOURCES,
        profile=profile,
    )


def test_stage_schema_literals_are_pinned_cross_repo():
    # Gridbook pins the identical literals in
    # ``tests/test_nvfp4_activation_contract.py``; they are a cross-repo ABI,
    # not a local constant.  The v1 literal must stay put because it also
    # frames ``target_values_sha256``.
    assert NVFP4_ACTIVATION_CONTRACT_SCHEMA == (
        "prismaquant.nvfp4_w4a4_activation.v1"
    )
    assert NVFP4_ACTIVATION_CONTRACT_SCHEMA_V2 == (
        "prismaquant.nvfp4_w4a4_activation.v2"
    )
    assert NVFP4_ROUTED_MOE_STAGE_SCHEMA == (
        "prismaquant.nvfp4_w4a4_activation_stages.v1"
    )
    assert NVFP4_ROUTED_MOE_STAGE_KEY == "routed_moe_stages"


def test_routed_moe_stage_names_only_packed_fusedmoe_stage_targets():
    assert routed_moe_stage(_W13) == ("model.layers.0.mlp.experts", "w13")
    assert routed_moe_stage(_W2) == ("model.layers.0.mlp.experts", "w2")
    # A dense MLP projection spells the same leaf but is not a routed stage.
    assert routed_moe_stage("model.layers.0.mlp.down_proj") is None
    # The per-expert split form is a Linear, not a FusedMoE stage.
    assert routed_moe_stage("model.layers.0.mlp.experts.7.gate_proj") is None
    assert routed_moe_stage("model.layers.0.self_attn.q_proj") is None
    # Profile-declared leaf naming resolves the same two stages.
    profile = _PackedExpertProfile()
    assert routed_moe_stage(
        "model.layers.0.mlp.experts.w1", profile=profile
    ) == ("model.layers.0.mlp.experts", "w13")
    assert routed_moe_stage(
        "model.layers.0.mlp.experts.w2", profile=profile
    ) == ("model.layers.0.mlp.experts", "w2")


def test_packed_contract_bumps_schema_and_dense_stays_v1_byte_for_byte():
    dense_record, _ = build_execution_contract(
        {"layer.q_proj": 2.0},
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        calibration_sources={"layer.q_proj": CALIBRATION_SOURCE_TARGET_CACHE},
    )
    assert dense_record["schema"] == NVFP4_ACTIVATION_CONTRACT_SCHEMA
    assert NVFP4_ROUTED_MOE_STAGE_KEY not in dense_record
    # A dense-only artifact is bit-identical to the pre-K0.2 record.
    legacy_record, _ = build_execution_contract(
        {"layer.q_proj": 2.0},
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    )
    assert dense_record == legacy_record

    record, physical = build_execution_contract(
        {_W13: 1.0, _W2: 2.0, "layer.q_proj": 2.0},
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        calibration_sources={
            **_STAGE_SOURCES,
            "layer.q_proj": CALIBRATION_SOURCE_TARGET_CACHE,
        },
    )
    assert record["schema"] == NVFP4_ACTIVATION_CONTRACT_SCHEMA_V2
    section = record[NVFP4_ROUTED_MOE_STAGE_KEY]
    assert section["schema"] == NVFP4_ROUTED_MOE_STAGE_SCHEMA
    assert section["module_names"] == ["model.layers.0.mlp.experts"]
    assert section["module_count"] == 1
    module = section["modules"]["model.layers.0.mlp.experts"]
    assert list(module) == ["w13", "w2"]
    assert module["w13"]["target"] == _W13
    assert module["w2"]["target"] == _W2
    assert module["w2"]["calibration_source"] == (
        CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY
    )
    # The whole-model digest fields are exactly what a pre-K0.2 reader
    # computed: the record-schema bump must not move them.
    assert record["target_values_sha256"] == target_values_sha256(
        physical, policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY
    )
    assert record["target_names"] == sorted((_W13, _W2, "layer.q_proj"))


def test_stage_digests_are_independent_per_stage():
    base = _stage_section()
    moved = _stage_section({_W13: 1.0, _W2: 4.0})
    base_modules = base["modules"]["model.layers.0.mlp.experts"]
    moved_modules = moved["modules"]["model.layers.0.mlp.experts"]
    assert base_modules["w13"]["stage_values_sha256"] == (
        moved_modules["w13"]["stage_values_sha256"]
    )
    assert base_modules["w2"]["stage_values_sha256"] != (
        moved_modules["w2"]["stage_values_sha256"]
    )
    assert base["stages_sha256"] != moved["stages_sha256"]
    # The calibration source is attested, not decorative.
    resourced = _stage_section(sources={
        **_STAGE_SOURCES,
        _W2: CALIBRATION_SOURCE_SUPPLEMENTAL_MAX_ABS,
    })
    assert resourced["modules"]["model.layers.0.mlp.experts"]["w2"][
        "stage_values_sha256"
    ] != base_modules["w2"]["stage_values_sha256"]


def test_half_attested_routed_moe_module_fails_closed():
    with pytest.raises(ValueError, match="both w13 and w2"):
        _stage_section({_W13: 1.0}, sources={_W13: _STAGE_SOURCES[_W13]})
    with pytest.raises(ValueError, match="both w13 and w2"):
        _stage_section({_W2: 2.0}, sources={_W2: _STAGE_SOURCES[_W2]})
    with pytest.raises(ValueError, match="no attested calibration source"):
        _stage_section(sources={_W13: _STAGE_SOURCES[_W13]})
    # w2 may never be calibrated from the experts-module input, and w13 may
    # never be calibrated from a routed-intermediate replay.
    with pytest.raises(ValueError, match="not a legal input for that stage"):
        _stage_section(sources={
            **_STAGE_SOURCES,
            _W2: CALIBRATION_SOURCE_PARENT_MODULE_CACHE,
        })
    with pytest.raises(ValueError, match="not a legal input for that stage"):
        _stage_section(sources={
            **_STAGE_SOURCES,
            _W13: CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
        })
    # An exporter that cannot attest sources at all must not claim readiness.
    with pytest.raises(ValueError, match="require calibration-source"):
        build_execution_contract(
            {_W13: 1.0, _W2: 2.0},
            policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        )
    # A physical mapper that renames the stage away fails closed too.
    with pytest.raises(ValueError, match="does not spell a packed FusedMoE"):
        build_execution_contract(
            {_W13: 1.0, _W2: 2.0},
            policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
            calibration_sources=_STAGE_SOURCES,
            target_name=lambda name: name.replace(".experts.", ".merged."),
        )


def test_sibling_unification_never_joins_across_stages():
    joined = unify_fused_sibling_input_global_scales({_W13: 0.5, _W2: 0.25})
    assert joined == {_W13: 0.5, _W2: 0.25}
    groups = group_fused_sibling_targets([_W13, _W2])
    assert sorted(map(tuple, groups.values())) == sorted([(_W13,), (_W2,)])


def test_stage_digest_framing_has_pinned_vector():
    assert stage_values_sha256(
        stage="w13",
        target="m.experts.gate_up_proj",
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        calibration_source=CALIBRATION_SOURCE_PARENT_MODULE_CACHE,
        value=1.0,
    ) == "c15c44ac3c290d4e596967218b41ffba2f12a857a2cc2356a1ed4e159a40e630"
    assert stage_values_sha256(
        stage="w2",
        target="m.experts.down_proj",
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        calibration_source=CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
        value=2.0,
    ) == "91f005ef177c3c8ccfb1f25a528d0a9a601ef4bdde61db8d33947a1a951cfe2e"
    section = build_routed_moe_stage_attestation(
        {"m.experts.gate_up_proj": 1.0, "m.experts.down_proj": 2.0},
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        calibration_sources={
            "m.experts.gate_up_proj": CALIBRATION_SOURCE_PARENT_MODULE_CACHE,
            "m.experts.down_proj": (
                CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY
            ),
        },
    )
    assert section["stages_sha256"] == (
        "77c830f2b1989a9a0069dcc7afabbe73f0913ccbfb634287346a0c097e231882"
    )
    # A stage digest is rooted at the stage schema and can never collide with
    # a whole-model digest over the same name/value pairs.
    assert section["stages_sha256"] != target_values_sha256(
        {"m.experts.gate_up_proj": 1.0, "m.experts.down_proj": 2.0},
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    )


def test_all_three_emit_paths_build_identical_stage_sections():
    from prismaquant.export_native_compressed import (
        _packed_expert_stage_attestation,
    )
    from prismaquant.production_weight_cache import ProductionWeightCache

    max_abs = {_W13: 3.0, _W2: 1.5}
    scales = {
        name: input_global_scale_from_max_abs(
            value, policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY
        )
        for name, value in max_abs.items()
    }
    sources = {name: CALIBRATION_SOURCE_PACKED_EXPERT_RENDER for name in scales}
    # Both CB exporters reach the same builder through
    # ``build_execution_contract``; the resident path maps logical -> physical
    # via the resolved skeleton name and the streaming path via its export
    # base name.  On these names both mappers are the identity.
    resident, _ = build_execution_contract(
        scales,
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        calibration_sources=sources,
        target_name=lambda name: name,
    )
    streaming, _ = build_execution_contract(
        scales,
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        calibration_sources=sources,
        target_name=lambda name: str(name),
    )
    cache = ProductionWeightCache(
        weights={}, levers={}, activation_max_abs=dict(max_abs)
    )
    native = _packed_expert_stage_attestation(_W2, cache=cache)
    assert json.dumps(resident[NVFP4_ROUTED_MOE_STAGE_KEY], sort_keys=True) == (
        json.dumps(streaming[NVFP4_ROUTED_MOE_STAGE_KEY], sort_keys=True)
    )
    assert json.dumps(native, sort_keys=True) == json.dumps(
        resident[NVFP4_ROUTED_MOE_STAGE_KEY], sort_keys=True
    )


def test_native_container_refuses_a_half_calibrated_fusedmoe_module():
    from prismaquant.export_native_compressed import (
        _packed_expert_stage_attestation,
    )
    from prismaquant.production_weight_cache import ProductionWeightCache

    cache = ProductionWeightCache(
        weights={}, levers={}, activation_max_abs={_W13: 3.0}
    )
    with pytest.raises(RuntimeError, match=r"no calibrated \['w2'\] stage"):
        _packed_expert_stage_attestation(_W13, cache=cache)
    # A dense projection carries no stage claim at all.
    assert _packed_expert_stage_attestation(
        "model.layers.0.mlp.down_proj", cache=cache
    ) is None


def _routed_moe_checkpoint(tmp_path):
    """Minimal per-expert MoE checkpoint plus its experts-module act entry."""

    hidden, inter, experts = 16, 8, 2
    model_dir = tmp_path / "moe"
    act_dir = tmp_path / "moe_act"
    model_dir.mkdir()
    generator = torch.Generator().manual_seed(19)
    tensors = {
        "model.layers.0.mlp.gate.weight": torch.randn(
            experts, hidden, generator=generator
        ),
    }
    for expert in range(experts):
        for leaf in ("gate_proj", "up_proj"):
            tensors[
                f"model.layers.0.mlp.experts.{expert}.{leaf}.weight"
            ] = torch.randn(inter, hidden, generator=generator)
    save_file(tensors, str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(json.dumps(
        {"num_experts_per_tok": 1, "norm_topk_prob": True}
    ))
    _write_activation(
        act_dir,
        "model.layers.0.mlp.experts",
        torch.randn(32, hidden, generator=generator),
    )
    return model_dir, act_dir


def test_routed_replay_and_module_input_are_distinct_attested_sources(tmp_path):
    from prismaquant.moe_imatrix import (
        synthesize_packed_expert_activation_samples,
    )

    model_dir, act_dir = _routed_moe_checkpoint(tmp_path)
    profile = _PackedExpertProfile()
    supplemental = synthesize_packed_expert_activation_samples(
        model_dir,
        act_dir,
        {_W13, _W2},
        profile,
        device="cpu",
    )
    scales, sources = calibrated_input_global_scales_with_sources(
        [_W13, _W2],
        activation_cache_dir=act_dir,
        policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        profile=profile,
        supplemental_activations=supplemental,
        calibration_device="cpu",
    )
    assert sources == {
        _W13: CALIBRATION_SOURCE_SUPPLEMENTAL_MODULE_INPUT,
        _W2: CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
    }
    # Two stages, two tensors, two values: the w2 scale is fitted on the
    # routed intermediate, so it is not the module-input scale.
    assert scales[_W13] != scales[_W2]
    record, _ = build_execution_contract(
        scales,
        policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        calibration_sources=sources,
        profile=profile,
    )
    assert record["schema"] == NVFP4_ACTIVATION_CONTRACT_SCHEMA_V2
    module = record[NVFP4_ROUTED_MOE_STAGE_KEY]["modules"][
        "model.layers.0.mlp.experts"
    ]
    assert module["w2"]["calibration_source"] == (
        CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY
    )
    assert module["w13"]["calibration_source"] == (
        CALIBRATION_SOURCE_SUPPLEMENTAL_MODULE_INPUT
    )


def test_missing_routed_intermediate_still_fails_closed(tmp_path):
    _model_dir, act_dir = _routed_moe_checkpoint(tmp_path)
    # Only the experts-module input is cached: w13 resolves from the parent
    # entry, w2 has no calibrated input at all.
    scales, sources = calibrated_input_global_scales_with_sources(
        [_W13],
        activation_cache_dir=act_dir,
        policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
    )
    assert sources == {_W13: CALIBRATION_SOURCE_PARENT_MODULE_CACHE}
    assert set(scales) == {_W13}
    with pytest.raises(ValueError, match="no calibrated input"):
        calibrated_input_global_scales(
            [_W13, _W2],
            activation_cache_dir=act_dir,
            policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        )
