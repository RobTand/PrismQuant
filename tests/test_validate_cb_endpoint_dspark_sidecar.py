from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import prismaquant.artifact_completeness as artifact_completeness
from prismaquant.gridbook_serving_runtime_pin import (
    GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA,
)
import prismaquant.validate_cb_endpoint as cbv
from prismaquant.cb_export_config import (
    _two_tier_scale_coding,
    source_passthrough_config_group,
    source_passthrough_wire_id,
)
from prismaquant.cb_layout import (
    SCALE_CODING_TWO_TIER,
    codebook_subtable_shapes,
    parse_format_name,
    type_size,
)
from prismaquant.dspark_source_metadata import (
    _released_dspark_tensor_layout,
    build_dspark_target_bridge,
    dspark_cb_construction_target_for_physical_output,
)
from prismaquant.nvfp4_activation_contract import (
    NVFP4_ACTIVATION_CONTRACT_KEY,
    NVFP4_ACTIVATION_CONTRACT_SCHEMA,
    NVFP4_ACTIVATION_EXECUTION,
    NVFP4_INPUT_GLOBAL_SCALE_SUFFIX,
)


_CONFIG = {
    "model_type": "deepseek_v4",
    "architectures": ["DeepseekV4ForCausalLM"],
    "hidden_size": 4096,
    "num_attention_heads": 64,
    "head_dim": 512,
    "q_lora_rank": 1024,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "vocab_size": 129280,
    "dspark_markov_rank": 256,
    "dspark_target_layer_ids": [40, 41, 42],
    "hc_mult": 4,
    "num_hidden_layers": 43,
    "n_mtp_layers": 3,
    "num_nextn_predict_layers": 1,
    "n_shared_experts": 1,
    "expert_dtype": "fp4",
    "dspark_block_size": 5,
}

_TARGET_TAILS = (
    "attn.wkv",
    "attn.wo_b",
    "attn.wq_a",
    "attn.wq_b",
    "ffn.experts.down_proj",
    "ffn.experts.gate_up_proj",
    "ffn.shared_experts.w1",
    "ffn.shared_experts.w2",
    "ffn.shared_experts.w3",
)
_PHYSICAL_TARGETS = tuple(sorted(
    f"mtp.{stage}.{tail}"
    for stage in range(3)
    for tail in _TARGET_TAILS
))
_CONSTRUCTION_TARGETS = tuple(sorted(
    f"model.layers.{43 + stage}.{tail}"
    for stage in range(3)
    for tail in _TARGET_TAILS
))
_SOURCE_MAPPING = dict(sorted({
    "mtp.0.main_proj": "model.main_proj",
    "mtp.0.attn.wo_a": "model.layers.43.attn.wo_a",
    "mtp.1.attn.wo_a": "model.layers.44.attn.wo_a",
    "mtp.2.attn.wo_a": "model.layers.45.attn.wo_a",
}.items()))

# These are the non-decoder tensors the released DSpark loader consumes.  Keep
# the names explicit: adding a source plane to this list would make the test
# fail to prove the 47-tensor glue closure requested by the endpoint gate.
_RETAINED_GLUE = (
    "mtp.0.attn.attn_sink",
    "mtp.0.attn.kv_norm.weight",
    "mtp.0.attn.q_norm.weight",
    "mtp.0.attn_norm.weight",
    "mtp.0.ffn.gate.bias",
    "mtp.0.ffn.gate.weight",
    "mtp.0.ffn_norm.weight",
    "mtp.0.hc_attn_base",
    "mtp.0.hc_attn_fn",
    "mtp.0.hc_attn_scale",
    "mtp.0.hc_ffn_base",
    "mtp.0.hc_ffn_fn",
    "mtp.0.hc_ffn_scale",
    "mtp.0.main_norm.weight",
    "mtp.1.attn.attn_sink",
    "mtp.1.attn.kv_norm.weight",
    "mtp.1.attn.q_norm.weight",
    "mtp.1.attn_norm.weight",
    "mtp.1.ffn.gate.bias",
    "mtp.1.ffn.gate.weight",
    "mtp.1.ffn_norm.weight",
    "mtp.1.hc_attn_base",
    "mtp.1.hc_attn_fn",
    "mtp.1.hc_attn_scale",
    "mtp.1.hc_ffn_base",
    "mtp.1.hc_ffn_fn",
    "mtp.1.hc_ffn_scale",
    "mtp.2.attn.attn_sink",
    "mtp.2.attn.kv_norm.weight",
    "mtp.2.attn.q_norm.weight",
    "mtp.2.attn_norm.weight",
    "mtp.2.confidence_head.proj.weight",
    "mtp.2.ffn.gate.bias",
    "mtp.2.ffn.gate.weight",
    "mtp.2.ffn_norm.weight",
    "mtp.2.hc_attn_base",
    "mtp.2.hc_attn_fn",
    "mtp.2.hc_attn_scale",
    "mtp.2.hc_ffn_base",
    "mtp.2.hc_ffn_fn",
    "mtp.2.hc_ffn_scale",
    "mtp.2.hc_head_base",
    "mtp.2.hc_head_fn",
    "mtp.2.hc_head_scale",
    "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.markov_w2.weight",
    "mtp.2.norm.weight",
)

_QWEIGHT_SHAPE_BY_TAIL = {
    "attn.wkv": [512, 912],
    "attn.wo_b": [4096, 1824],
    "attn.wq_a": [1024, 912],
    "attn.wq_b": [32768, 228],
    "ffn.experts.down_proj": [256, 4096, 456],
    "ffn.experts.gate_up_proj": [256, 4096, 912],
    "ffn.shared_experts.w1": [2048, 912],
    "ffn.shared_experts.w2": [4096, 456],
    "ffn.shared_experts.w3": [2048, 912],
}
_CODEBOOK_REFS = (
    "cb_codebook.lattice.NVFP4_CB_K12.sub0",
    "cb_codebook.lattice.NVFP4_CB_K12.sub1",
)


def _scheme() -> dict:
    return {
        "act_bits": 4,
        "codebook_group": None,
        "codebook_ref": list(_CODEBOOK_REFS),
        "codebook_source": "lattice",
        "grid": "fp4",
        "group_size": 16,
        "k": 12,
        "mode": "product",
        "n_sub": 2,
        "scale_coding": _two_tier_scale_coding(),
        "superblock": 256,
        "type_size": 57,
        "vec_dim": 8,
    }


def _codebook_tensors() -> dict[str, torch.Tensor]:
    return {
        _CODEBOOK_REFS[0]: torch.zeros((64, 4), dtype=torch.float16),
        _CODEBOOK_REFS[1]: torch.ones((64, 4), dtype=torch.float16),
    }


def _codebook_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.to(torch.float16).cpu().numpy().tobytes()
    ).hexdigest()


def _finalized_tensor_formats() -> dict[str, str]:
    formats: dict[str, str] = {}
    for stage in range(3):
        prefix = f"mtp.{stage}."
        for tail in _TARGET_TAILS:
            if not tail.startswith("ffn.experts."):
                formats[prefix + tail] = "NVFP4_CB_K12"
        for expert in range(_CONFIG["n_routed_experts"]):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                formats[
                    f"{prefix}ffn.experts.{expert}.{projection}"
                ] = "NVFP4_CB_K12"
    formats.update({
        target: "FP8_BLOCK_UE8M0_SOURCE" for target in _SOURCE_MAPPING
    })
    return dict(sorted(formats.items()))


def _quant_config() -> dict:
    source_group = source_passthrough_config_group(
        "FP8_BLOCK_UE8M0_SOURCE"
    )
    source_group["targets"] = [
        "re:^mtp[.]0[.]attn[.]wo_a$",
        "re:^mtp[.]0[.]main_proj$",
        "re:^mtp[.]1[.]attn[.]wo_a$",
        "re:^mtp[.]2[.]attn[.]wo_a$",
    ]
    sidecar = {
        "schema": cbv.DSPARK_CB_SIDECAR_SCHEMA,
        "num_hidden_layers": 43,
        "n_mtp_layers": 3,
        "physical_namespace": "mtp.{stage}",
        "construction_namespace": "model.layers.{num_hidden_layers+stage}",
        "physical_cb_targets": list(_PHYSICAL_TARGETS),
        "construction_cb_targets": list(_CONSTRUCTION_TARGETS),
        "source_passthrough_targets": sorted(_SOURCE_MAPPING),
        "source_passthrough_physical_to_construction": _SOURCE_MAPPING,
        "activation_bridge_present": False,
    }
    return {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            "group_0": {
                "format": "NVFP4_CB_K12",
                "scheme": _scheme(),
                "targets": list(_CONSTRUCTION_TARGETS),
            },
            "group_1": source_group,
        },
        "source_passthrough": {
            "version": 1,
            "units": {
                construction: source_passthrough_wire_id(
                    "FP8_BLOCK_UE8M0_SOURCE"
                )
                for construction in _SOURCE_MAPPING.values()
            },
        },
        "provenance": {
            "weight_content_manifest": {"schema": "test"},
            "dspark_cb_sidecar": sidecar,
            "serialized_payload": {
                "n_tensors": 27,
                "codebook_sidecar_bytes": 1024,
                "sidecars": [{"codebook_ref": list(_CODEBOOK_REFS)}],
            },
            "codebook_sha256": {
                ref: _codebook_digest(tensor)
                for ref, tensor in _codebook_tensors().items()
            },
            "tensor_formats": _finalized_tensor_formats(),
        },
    }


def _released_layout():
    return _released_dspark_tensor_layout(
        hidden_size=4096,
        num_heads=64,
        head_dim=512,
        q_lora_rank=1024,
        o_groups=8,
        o_lora_rank=1024,
        moe_intermediate_size=2048,
        num_experts=256,
        vocab_size=129280,
        markov_rank=256,
        target_layer_count=3,
        hc_mult=4,
        fp8_block=(128, 128),
    )[0]


def _model_header() -> dict[str, dict]:
    layout = _released_layout()
    header = {
        name: {
            "dtype": sorted(layout[name].dtypes)[0],
            "shape": list(layout[name].shape),
            "data_offsets": [0, 1],
        }
        for name in _RETAINED_GLUE
    }
    for target in _PHYSICAL_TARGETS:
        tail = ".".join(target.split(".")[2:])
        header[target + ".cb_qweight"] = {
            "dtype": "U8",
            "shape": list(_QWEIGHT_SHAPE_BY_TAIL[tail]),
            "data_offsets": [0, 1],
        }
    for base in _SOURCE_MAPPING:
        for suffix in ("weight", "scale"):
            name = base + "." + suffix
            contract = layout[name]
            header[name] = {
                "dtype": sorted(contract.dtypes)[0],
                "shape": list(contract.shape),
                "data_offsets": [0, 1],
            }
    return header


def _codebook_header() -> dict[str, dict]:
    return {
        _CODEBOOK_REFS[0]: {
            "dtype": "F16",
            "shape": [64, 4],
            "data_offsets": [0, 512],
        },
        _CODEBOOK_REFS[1]: {
            "dtype": "F16",
            "shape": [64, 4],
            "data_offsets": [512, 1024],
        },
    }


@dataclass(frozen=True)
class _Complete:
    ok: bool = True
    declared_units: tuple[str, ...] = tuple(_SOURCE_MAPPING.values())
    cb_units: tuple[str, ...] = tuple(_CONSTRUCTION_TARGETS)
    passthrough_units: tuple[str, ...] = tuple(_SOURCE_MAPPING.values())
    verbatim_namespace_units: tuple[str, ...] = ()
    route_pending_acknowledged: tuple[str, ...] = ()
    excluded_namespaces: tuple[str, ...] = ()


@pytest.fixture
def sidecar_case(monkeypatch, tmp_path: Path):
    config = deepcopy(_CONFIG)
    quant = _quant_config()
    header = _model_header()
    codebooks = _codebook_header()
    save_file(_codebook_tensors(), str(tmp_path / "cb_codebooks.pqcb"))
    monkeypatch.setattr(
        artifact_completeness,
        "read_artifact_header",
        lambda _root: deepcopy(header),
    )
    monkeypatch.setattr(
        artifact_completeness,
        "_read_safetensors_header",
        lambda _path: deepcopy(codebooks),
    )
    monkeypatch.setattr(
        artifact_completeness,
        "assert_artifact_complete",
        lambda *_args, **_kwargs: _Complete(),
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path, config, quant, header


def _validate(case):
    root, config, quant, _header = case
    return cbv._validate_dspark_cb_sidecar_artifact(
        root,
        model_config=config,
        quant_config=quant,
    )


def test_sidecar_closes_exact_27_targets_47_glue_and_8_source_planes(
    sidecar_case,
):
    root, config, quant, header = sidecar_case
    assert len(_PHYSICAL_TARGETS) == 27
    assert len(_CONSTRUCTION_TARGETS) == 27
    assert len(_RETAINED_GLUE) == cbv.DSPARK_CB_RETAINED_GLUE_TENSOR_COUNT == 47

    payload = _validate(sidecar_case)

    assert payload["provenance"]["physical_cb_targets"] == list(
        _PHYSICAL_TARGETS
    )
    assert payload["provenance"]["construction_cb_targets"] == list(
        _CONSTRUCTION_TARGETS
    )
    assert payload["physical_to_construction"] == {
        physical: construction
        for physical, construction in zip(
            _PHYSICAL_TARGETS, _CONSTRUCTION_TARGETS, strict=True
        )
    }
    assert set(payload["header_contract"]) == set(header)
    assert set(_RETAINED_GLUE).issubset(payload["header_contract"])
    assert {
        base + suffix
        for base in _SOURCE_MAPPING
        for suffix in (".weight", ".scale")
    }.issubset(payload["header_contract"])
    assert cbv.DSPARK_CB_HYBRID_WEIGHT_ONLY_TENSOR_COUNT == 82
    assert len(payload["header_contract"]) == 82
    assert not any(
        name.endswith((".weight", ".scale"))
        and name.startswith(tuple(target + "." for target in _PHYSICAL_TARGETS))
        for name in header
        if not name.startswith("mtp.0.main_proj.")
    )

    # The same fixture must also build a self-hashed v2 receipt whose runtime
    # feature requirement is explicit rather than inferred from a version.
    receipt = cbv.validate_cb_artifact_decode_contract(
        root,
        quant,
        runtime_pin={
            "runtime_contract_schema":
                GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA,
            "required_abi_features": {
                cbv.DSPARK_CB_RUNTIME_FEATURE:
                    cbv.DSPARK_CB_RUNTIME_FEATURE_VERSION,
                cbv.DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE:
                    cbv.DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE_VERSION,
            }
        },
    )
    assert receipt["schema"] == cbv.ARTIFACT_DECODE_CONTRACT_SCHEMA_V2
    assert receipt["required_runtime_features"] == {
        cbv.DSPARK_CB_RUNTIME_FEATURE:
            cbv.DSPARK_CB_RUNTIME_FEATURE_VERSION,
        cbv.DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE:
            cbv.DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE_VERSION,
    }
    assert receipt["required_runtime_contract_schema"] == (
        GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA
    )
    cbv._validate_artifact_decode_record(receipt)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
@pytest.mark.parametrize("plane_class", ["glue", "qweight"])
def test_sidecar_rejects_missing_or_extra_glue_and_qweights(
    sidecar_case, mutation: str, plane_class: str
):
    _root, _config, _quant, header = sidecar_case
    if plane_class == "glue":
        name = _RETAINED_GLUE[0]
        extra_name = "mtp.1.unknown_glue.weight"
    else:
        name = _PHYSICAL_TARGETS[0] + ".cb_qweight"
        extra_name = "mtp.0.attn.unknown.cb_qweight"
    if mutation == "missing":
        header.pop(name)
    else:
        header[extra_name] = {
            "dtype": "U8" if plane_class == "qweight" else "BF16",
            "shape": [1],
            "data_offsets": [0, 1],
        }

    expected = "qweight headers" if plane_class == "qweight" else "tensor namespace"
    with pytest.raises(cbv.CBEndpointValidationError, match=expected):
        _validate(sidecar_case)


def test_sidecar_rejects_config_header_namespace_mismatch(sidecar_case):
    _root, _config, quant, _header = sidecar_case
    targets = quant["config_groups"]["group_0"]["targets"]
    targets[0] = _PHYSICAL_TARGETS[0]
    targets.sort()

    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="config groups do not exactly cover construction targets",
    ):
        _validate(sidecar_case)


def test_sidecar_rejects_format_label_without_matching_canonical_scheme(
    sidecar_case,
):
    _root, _config, quant, _header = sidecar_case
    quant["config_groups"]["group_0"]["format"] = "NVFP4_CB_K13"

    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="format/scheme layout disagree",
    ):
        _validate(sidecar_case)


def test_sidecar_rejects_self_consistent_format_scheme_and_header_rewrite_against_assignment(
    sidecar_case, monkeypatch,
):
    root, _config, quant, header = sidecar_case
    group = quant["config_groups"]["group_0"]
    group["format"] = "NVFP4_CB_K13"
    group["scheme"]["k"] = 13
    group["scheme"]["type_size"] = type_size(
        13, "fp4", SCALE_CODING_TWO_TIER
    )

    # Rewrite both the packed-weight geometry and the real safetensors
    # codebook to a completely self-consistent K13 layout.  The immutable
    # finalized per-Linear assignment remains K12 and must be the independent
    # authority that rejects this otherwise coherent mutation.
    old_type_size = type_size(12, "fp4", SCALE_CODING_TWO_TIER)
    new_type_size = type_size(13, "fp4", SCALE_CODING_TWO_TIER)
    for name, metadata in header.items():
        if name.endswith(".cb_qweight"):
            metadata["shape"][-1] = (
                metadata["shape"][-1] // old_type_size * new_type_size
            )
    parsed = parse_format_name("NVFP4_CB_K13")
    assert parsed is not None
    family, k = parsed
    shapes = codebook_subtable_shapes(k, family.mode, family.n_sub)
    rewritten_refs = tuple(
        ref.replace("NVFP4_CB_K12", "NVFP4_CB_K13")
        for ref in _CODEBOOK_REFS
    )
    group["scheme"]["codebook_ref"] = list(rewritten_refs)
    rewritten_codebooks = {
        ref: torch.full(shape, index, dtype=torch.float16)
        for index, (ref, shape) in enumerate(
            zip(rewritten_refs, shapes, strict=True), start=1
        )
    }
    save_file(rewritten_codebooks, str(root / "cb_codebooks.pqcb"))
    offset = 0
    rewritten_header = {}
    for ref, tensor in rewritten_codebooks.items():
        nbytes = tensor.numel() * tensor.element_size()
        rewritten_header[ref] = {
            "dtype": "F16",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    monkeypatch.setattr(
        artifact_completeness,
        "_read_safetensors_header",
        lambda _path: deepcopy(rewritten_header),
    )
    quant["provenance"]["serialized_payload"][
        "codebook_sidecar_bytes"
    ] = offset
    quant["provenance"]["serialized_payload"]["sidecars"][0][
        "codebook_ref"
    ] = list(rewritten_refs)
    quant["provenance"]["codebook_sha256"] = {
        ref: _codebook_digest(tensor)
        for ref, tensor in rewritten_codebooks.items()
    }

    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="config-group formats differ from finalized physical "
        "tensor_formats assignment",
    ):
        _validate(sidecar_case)


def test_sidecar_group_format_binds_every_expanded_expert_assignment_member(
    sidecar_case,
):
    _root, _config, quant, _header = sidecar_case
    last_expert_member = "mtp.2.ffn.experts.255.up_proj"
    quant["provenance"]["tensor_formats"][
        last_expert_member
    ] = "NVFP4_CB_K13"

    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="config-group formats differ from finalized physical "
        "tensor_formats assignment",
    ):
        _validate(sidecar_case)


@pytest.mark.parametrize("mutated_ref", _CODEBOOK_REFS)
def test_sidecar_rejects_real_safetensors_codebook_payload_mutation(
    sidecar_case, mutated_ref,
):
    root, _config, _quant, _header = sidecar_case
    tensors = _codebook_tensors()
    tensors[mutated_ref][0, 0] += torch.tensor(1, dtype=torch.float16)
    save_file(tensors, str(root / "cb_codebooks.pqcb"))

    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="logical FP16 tensor payload SHA-256 differs",
    ):
        _validate(sidecar_case)


def test_sidecar_rejects_source_plane_for_cb_decoder_target(sidecar_case):
    _root, _config, _quant, header = sidecar_case
    source = _released_layout()["mtp.0.attn.wq_a.weight"]
    header["mtp.0.attn.wq_a.weight"] = {
        "dtype": sorted(source.dtypes)[0],
        "shape": list(source.shape),
        "data_offsets": [0, 1],
    }

    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="source planes for CB decoder targets",
    ):
        _validate(sidecar_case)


@pytest.mark.parametrize("mutation", ["missing", "dtype", "shape"])
def test_sidecar_rejects_missing_or_malformed_wo_a_source_plane(
    sidecar_case, mutation: str
):
    _root, _config, _quant, header = sidecar_case
    name = "mtp.1.attn.wo_a.weight"
    if mutation == "missing":
        header.pop(name)
    elif mutation == "dtype":
        header[name]["dtype"] = "BF16"
    else:
        header[name]["shape"] = [1, 1]

    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="tensor namespace|hybrid W8A16 source plane",
    ):
        _validate(sidecar_case)


def test_sidecar_rejects_wo_a_declared_as_cb_and_source(sidecar_case):
    _root, _config, quant, header = sidecar_case
    physical = "mtp.0.attn.wo_a"
    construction = "model.layers.43.attn.wo_a"
    quant["provenance"]["dspark_cb_sidecar"]["physical_cb_targets"].append(
        physical
    )
    quant["provenance"]["dspark_cb_sidecar"]["physical_cb_targets"].sort()
    quant["provenance"]["dspark_cb_sidecar"][
        "construction_cb_targets"
    ].append(construction)
    quant["provenance"]["dspark_cb_sidecar"][
        "construction_cb_targets"
    ].sort()
    quant["config_groups"]["group_0"]["targets"].append(construction)
    quant["config_groups"]["group_0"]["targets"].sort()
    header[physical + ".cb_qweight"] = {
        "dtype": "U8",
        "shape": [8192, 912],
        "data_offsets": [0, 1],
    }

    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="exact three-stage decoder target set",
    ):
        _validate(sidecar_case)


@pytest.mark.parametrize("state", ["bridge_only", "activation_only"])
def test_sidecar_rejects_bridge_activation_xor(sidecar_case, state: str):
    _root, config, quant, _header = sidecar_case
    activation = {
        "schema": NVFP4_ACTIVATION_CONTRACT_SCHEMA,
        "contract": NVFP4_ACTIVATION_EXECUTION,
        "group_size": 16,
        "tensor_suffix": NVFP4_INPUT_GLOBAL_SCALE_SUFFIX,
        "value_dtype": "float32",
        "target_names": [
            "mtp.0.attn.wq_a",
            "mtp.1.attn.wq_a",
            "mtp.2.attn.wq_a",
        ],
        "target_count": 3,
        "target_values_sha256": "a" * 64,
    }
    construction = [
        dspark_cb_construction_target_for_physical_output(target, config)
        for target in activation["target_names"]
    ]
    bridge = build_dspark_target_bridge(
        config,
        contracted_cb_construction_targets=construction,
        activation_execution_contract=activation,
    )
    if state == "bridge_only":
        quant["dspark_target_bridge"] = bridge
    else:
        quant["execution_contracts"] = {
            NVFP4_ACTIVATION_CONTRACT_KEY: activation
        }
        quant["provenance"]["dspark_cb_sidecar"][
            "activation_bridge_present"
        ] = True

    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="wholly present or wholly absent",
    ):
        _validate(sidecar_case)


def test_sidecar_accepts_matching_bridge_and_activation_contract(sidecar_case):
    _root, config, quant, header = sidecar_case
    physical = [
        "mtp.0.attn.wq_a",
        "mtp.1.attn.wq_a",
        "mtp.2.attn.wq_a",
    ]
    construction = [
        dspark_cb_construction_target_for_physical_output(target, config)
        for target in physical
    ]
    activation = {
        "schema": NVFP4_ACTIVATION_CONTRACT_SCHEMA,
        "contract": NVFP4_ACTIVATION_EXECUTION,
        "group_size": 16,
        "tensor_suffix": NVFP4_INPUT_GLOBAL_SCALE_SUFFIX,
        "value_dtype": "float32",
        "target_names": physical,
        "target_count": 3,
        "target_values_sha256": "a" * 64,
    }
    base_targets = quant["config_groups"]["group_0"]["targets"]
    quant["config_groups"]["group_0"]["targets"] = sorted(
        set(base_targets) - set(construction)
    )
    activation_scheme = _scheme()
    activation_scheme["activation_contract"] = NVFP4_ACTIVATION_CONTRACT_KEY
    quant["config_groups"]["group_activation"] = {
        "format": "NVFP4_CB_K12",
        "scheme": activation_scheme,
        "targets": construction,
    }
    quant["execution_contracts"] = {
        NVFP4_ACTIVATION_CONTRACT_KEY: activation
    }
    quant["dspark_target_bridge"] = build_dspark_target_bridge(
        config,
        contracted_cb_construction_targets=construction,
        activation_execution_contract=activation,
    )
    quant["provenance"]["dspark_cb_sidecar"][
        "activation_bridge_present"
    ] = True
    for target in physical:
        header[target + ".input_global_scale"] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        }

    payload = _validate(sidecar_case)

    assert payload["activation_execution_contract"] == activation
    assert payload["dspark_target_bridge"] == quant["dspark_target_bridge"]


@pytest.mark.parametrize("provenance_state", ["both", "neither"])
def test_decode_contract_requires_overlay_sidecar_provenance_xor(
    sidecar_case, provenance_state: str
):
    root, _config, quant, _header = sidecar_case
    if provenance_state == "both":
        quant["provenance"]["dspark_source_overlay"] = {"schema": "test"}
    else:
        quant["provenance"].pop("dspark_cb_sidecar")

    # Both directions still refuse; they say different things because they are
    # different faults. Declaring both is a contradiction on any lane, while
    # declaring neither is only a fault on the lane whose serving stack needs
    # one of the two bridges -- or, since the split release, a declaration that
    # the artifact is the body half, which this fixture does not make.
    expected = (
        "declares both dspark_source_overlay and dspark_cb_sidecar"
        if provenance_state == "both"
        else "one of dspark_source_overlay or dspark_cb_sidecar"
    )
    with pytest.raises(cbv.CBEndpointValidationError, match=expected):
        cbv.validate_cb_artifact_decode_contract(root, quant)


@pytest.mark.parametrize(
    "missing_feature",
    [
        cbv.DSPARK_CB_RUNTIME_FEATURE,
        cbv.DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE,
    ],
)
def test_sidecar_receipt_rejects_runtime_without_required_feature(
    sidecar_case, missing_feature: str
):
    root, _config, quant, _header = sidecar_case
    features = {
        cbv.DSPARK_CB_RUNTIME_FEATURE:
            cbv.DSPARK_CB_RUNTIME_FEATURE_VERSION,
        cbv.DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE:
            cbv.DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE_VERSION,
    }
    features.pop(missing_feature)
    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="does not implement the DSpark sidecar decode ABI",
    ):
        cbv.validate_cb_artifact_decode_contract(
            root,
            quant,
            runtime_pin={
                "runtime_contract_schema":
                    GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA,
                "required_abi_features": features,
            },
        )


def test_sidecar_receipt_rejects_runtime_contract_v3_even_with_feature(
    sidecar_case,
):
    root, _config, quant, _header = sidecar_case
    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="runtime-contract schema",
    ):
        cbv.validate_cb_artifact_decode_contract(
            root,
            quant,
            runtime_pin={
                "runtime_contract_schema": "gridbook.runtime-contract.v3",
                "required_abi_features": {
                    cbv.DSPARK_CB_RUNTIME_FEATURE:
                        cbv.DSPARK_CB_RUNTIME_FEATURE_VERSION,
                    cbv.DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE:
                        cbv.DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE_VERSION,
                },
            },
        )


def test_sidecar_rejects_incomplete_source_group_or_construction_units(
    sidecar_case,
):

    _root, _config, quant, _header = sidecar_case
    quant["config_groups"]["group_1"]["targets"].pop()
    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="exact block-128 source-FP8 W8A16 group",
    ):
        _validate(sidecar_case)

    replacement = _quant_config()
    replacement["source_passthrough"]["units"].pop(
        "model.layers.45.attn.wo_a"
    )
    replacement_case = (
        sidecar_case[0], sidecar_case[1], replacement, sidecar_case[3]
    )
    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="exactly the four W8A16",
    ):
        _validate(replacement_case)


def test_sidecar_receipt_rejects_stale_hybrid_source_mapping(sidecar_case):
    root, _config, quant, _header = sidecar_case
    receipt = cbv.validate_cb_artifact_decode_contract(root, quant)
    receipt["dspark_cb_sidecar"][
        "source_passthrough_physical_to_construction"
    ].pop("mtp.2.attn.wo_a")

    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="invalid topology or bridge state",
    ):
        cbv._validate_artifact_decode_record(receipt)
