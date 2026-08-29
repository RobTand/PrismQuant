"""CPU producer tests for PROPOSED per-expert Gridbook split stacks."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

os.environ["PRISMAQUANT_CB_ENCODE_COMPILE"] = "0"

from prismaquant.artifact_completeness import (  # noqa: E402
    ArtifactIncomplete,
    assert_artifact_complete,
)
from prismaquant import nvfp4_cb_formats as cb  # noqa: E402
from prismaquant.export_nvfp4_cb_streaming import (  # noqa: E402
    export_nvfp4_cb_streaming,
)
from prismaquant.layer_config import canonicalize_assignment  # noqa: E402
from prismaquant.shipcard import load_shipcard  # noqa: E402


HIDDEN = 256
EXPERTS = 4
CB16 = {"data_type": "nvfp4_cb", "cb_k": 16}
FP8_40 = {"data_type": "fp8_cb", "cb_k": 40}
MX_SOURCE = {"data_type": "fp4_e2m1", "bits": 4, "group_size": 32}
RECIPE_PREFIX = "model.layers.0.mlp.experts"
PHYSICAL_PREFIX = "layers.0.ffn.experts"


def _e8m0(shape, generator):
    return torch.randint(110, 140, shape, generator=generator).to(
        torch.uint8
    ).view(torch.float8_e8m0fnu)


def _write_source(root: Path):
    root.mkdir()
    generator = torch.Generator().manual_seed(91)
    tensors = {}
    for expert_id in range(EXPERTS):
        for leaf in ("w1", "w3", "w2"):
            base = f"{PHYSICAL_PREFIX}.{expert_id}.{leaf}"
            tensors[base + ".weight"] = torch.randint(
                -128, 128, (HIDDEN, HIDDEN // 2),
                dtype=torch.int8, generator=generator,
            )
            tensors[base + ".scale"] = _e8m0(
                (HIDDEN, HIDDEN // 32), generator
            )
    tensors["norm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN,
        "n_routed_experts": EXPERTS,
        "expert_dtype": "fp4",
        "quantization_config": {
            "quant_method": "fp8", "fmt": "e4m3",
            "weight_block_size": [128, 128], "scale_fmt": "ue8m0",
        },
    }))
    return tensors


def _flat_config(format_for):
    return {
        f"{RECIPE_PREFIX}.{expert_id}.{projection}": format_for(
            expert_id, projection
        )
        for expert_id in range(EXPERTS)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload))
    return path


def _col_weights():
    generator = torch.Generator().manual_seed(17)
    return {
        f"{RECIPE_PREFIX}.{expert_id}.{projection}": (
            torch.rand(HIDDEN, generator=generator) + 0.05
        )
        for expert_id in range(EXPERTS)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }


def _mixed_format(expert_id, projection):
    if projection in ("gate_proj", "up_proj"):
        return CB16 if expert_id < 2 else FP8_40
    return (CB16, FP8_40, MX_SOURCE, MX_SOURCE)[expert_id]


def _export(root: Path, *, per_expert=None, name="out"):
    source = root / "source"
    source_tensors = _write_source(source)
    base = _write_json(
        root / "base.json", _flat_config(lambda _expert, _projection: CB16)
    )
    per_path = None
    if per_expert is not None:
        per_path = _write_json(root / "per-expert.json", per_expert)
    out = root / name
    export_nvfp4_cb_streaming(
        source, base, out, _col_weights(), device="cpu",
        allow_unstamped_research=True,
        per_expert_config_path=per_path,
    )
    return source, source_tensors, out


@pytest.fixture(scope="module")
def mixed_export(tmp_path_factory):
    return _export(
        tmp_path_factory.mktemp("per-expert-mixed"),
        per_expert=_flat_config(_mixed_format),
    )


def test_mixed_layer_export_round_trip_and_exact_declaration(mixed_export):
    _source, _source_tensors, out = mixed_export
    tensors = load_file(str(out / "model.safetensors"))
    quant_config = json.loads((out / "quant_config.json").read_text())
    declaration = quant_config["per_expert_format_groups"]
    tensor_formats = quant_config["provenance"]["tensor_formats"]
    assert tensor_formats == canonicalize_assignment(_flat_config(_mixed_format))

    assert declaration == {
        "version": 1,
        "layers": {
            "0": {
                "w13": [
                    {
                        "format_wire_id": "FP8_CB_K40",
                        "expert_ids": [2, 3],
                        "tensor_prefix": (
                            f"{PHYSICAL_PREFIX}.gate_up_proj."
                            "format_group_fp8_cb_k40"
                        ),
                    },
                    {
                        "format_wire_id": "NVFP4_CB_K16",
                        "expert_ids": [0, 1],
                        "tensor_prefix": (
                            f"{PHYSICAL_PREFIX}.gate_up_proj."
                            "format_group_nvfp4_cb_k16"
                        ),
                    },
                ],
                "w2": [
                    {
                        "format_wire_id": "FP8_CB_K40",
                        "expert_ids": [1],
                        "tensor_prefix": (
                            f"{PHYSICAL_PREFIX}.down_proj."
                            "format_group_fp8_cb_k40"
                        ),
                    },
                    {
                        "format_wire_id": "NVFP4_CB_K16",
                        "expert_ids": [0],
                        "tensor_prefix": (
                            f"{PHYSICAL_PREFIX}.down_proj."
                            "format_group_nvfp4_cb_k16"
                        ),
                    },
                    {
                        "format_wire_id": "mxfp4_e2m1_ue8m0_g32",
                        "expert_ids": [2, 3],
                        "tensor_prefix": PHYSICAL_PREFIX,
                    },
                ],
            }
        },
    }
    # The first dimension is the declaration's ascending expert order.
    assert tensors[
        f"{PHYSICAL_PREFIX}.gate_up_proj.format_group_fp8_cb_k40.cb_qweight"
    ].shape[0] == 2
    assert tensors[
        f"{PHYSICAL_PREFIX}.down_proj.format_group_fp8_cb_k40.cb_qweight"
    ].shape[0] == 1
    codebooks = load_file(str(out / quant_config["codebook_file"]))
    schemes = {
        target: group["scheme"]
        for group in quant_config["config_groups"].values()
        if "scheme" in group
        for target in group["targets"]
    }
    # Real producer-byte round trip: declaration prefix -> config scheme ->
    # sidecar codebook -> unpack -> reassemble, for every CB subgroup.
    for families in declaration["layers"].values():
        for entries in families.values():
            for entry in entries:
                prefix = entry["tensor_prefix"]
                if prefix not in schemes:
                    continue
                scheme = schemes[prefix]
                packed = tensors[prefix + ".cb_qweight"]
                rows = packed.numel() // packed.shape[-1]
                refs = scheme["codebook_ref"]
                codebook = (
                    tuple(codebooks[name].float() for name in refs)
                    if isinstance(refs, list)
                    else codebooks[refs].float()
                )
                scales = tensors.get(prefix + ".weight_scale")
                coding = (
                    cb.SCALE_CODING_TWO_TIER
                    if "scale_coding" in scheme else cb.SCALE_CODING_V1
                )
                fields = cb.nvfp4_cb_unpack(
                    packed.reshape(rows, -1),
                    scheme["k"], scheme["grid"], scheme["mode"],
                    (rows, HIDDEN), codebook=codebook,
                    scales=(scales.reshape(rows, 1)
                            if scales is not None else None),
                    scale_coding=coding,
                )
                repacked = cb.nvfp4_cb_assemble_bytes(
                    fields, scheme["k"], scheme["grid"], scheme["mode"],
                )
                assert torch.equal(repacked.reshape_as(packed), packed)
    from prismaquant import footprint
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    allocation = canonicalize_assignment(_flat_config(_mixed_format))
    stats = {
        qname: {
            "n_params": HIDDEN * HIDDEN,
            "in_features": HIDDEN,
            "out_features": HIDDEN,
        }
        for qname in allocation
    }
    pre_export = footprint.per_expert_format_group_payload_breakdown(
        allocation,
        stats,
        context=CBSerializationContext(
            scale_coding="two_tier",
            codebook_source="lattice",
            scale_sweep=True,
        ),
    )
    emitted = quant_config["provenance"]["per_expert_format_group_payload"]
    assert pre_export["tensor_payload_bytes"] == emitted["tensor_payload_bytes"]
    assert pre_export["codebook_sidecar_bytes"] == emitted[
        "codebook_sidecar_bytes"
    ]
    assert pre_export["total_bytes"] == emitted["total_bytes"]
    assert_artifact_complete(out)


def test_mxfp4_subgroup_is_verbatim_and_not_double_declared(mixed_export):
    source, _source_tensors, out = mixed_export
    source_bytes = (source / "model.safetensors").read_bytes()
    output_bytes = (out / "model.safetensors").read_bytes()
    # Compare tensor payloads through safetensors rather than file offsets,
    # which necessarily move when CB substacks are added.
    source_values = load_file(str(source / "model.safetensors"))
    output_values = load_file(str(out / "model.safetensors"))
    for expert_id in (2, 3):
        for plane in ("weight", "scale"):
            name = f"{PHYSICAL_PREFIX}.{expert_id}.w2.{plane}"
            assert torch.equal(
                output_values[name].view(torch.uint8),
                source_values[name].view(torch.uint8),
            ), name
    assert hashlib.sha256(source_bytes).digest() != hashlib.sha256(output_bytes).digest()
    quant_config = json.loads((out / "quant_config.json").read_text())
    passthrough = (quant_config.get("source_passthrough") or {}).get("units", {})
    assert PHYSICAL_PREFIX not in passthrough
    assert RECIPE_PREFIX not in passthrough


@pytest.mark.parametrize("mutation, message", [
    ("missing", r"missing expert ids \[3\]"),
    ("duplicate", "duplicated expert id 1"),
    ("undeclared_tensor", "have no declaration"),
])
def test_completeness_refuses_broken_group_contract(
    mixed_export, tmp_path, mutation, message,
):
    _source, _source_tensors, healthy = mixed_export
    broken = tmp_path / mutation
    shutil.copytree(healthy, broken)
    quant_path = broken / "quant_config.json"
    quant_config = json.loads(quant_path.read_text())
    if mutation == "missing":
        quant_config["per_expert_format_groups"]["layers"]["0"]["w13"][0][
            "expert_ids"
        ].remove(3)
        quant_path.write_text(json.dumps(quant_config))
    elif mutation == "duplicate":
        quant_config["per_expert_format_groups"]["layers"]["0"]["w2"][0][
            "expert_ids"
        ].append(1)
        quant_path.write_text(json.dumps(quant_config))
    else:
        tensors = load_file(str(broken / "model.safetensors"))
        tensors[
            f"{PHYSICAL_PREFIX}.down_proj.format_group_ghost.cb_qweight"
        ] = torch.zeros((1, 1, 1), dtype=torch.uint8)
        save_file(tensors, str(broken / "model.safetensors"))
    with pytest.raises(ArtifactIncomplete, match=message):
        assert_artifact_complete(broken)


def test_uniform_per_expert_mode_is_byte_identical_to_legacy(tmp_path):
    source = tmp_path / "source"
    _write_source(source)
    mapping = _flat_config(lambda _expert, _projection: CB16)
    base = _write_json(tmp_path / "base.json", mapping)
    per_path = _write_json(tmp_path / "per.json", mapping)
    for out, per_config in (
        (tmp_path / "legacy", None),
        (tmp_path / "uniform", per_path),
    ):
        export_nvfp4_cb_streaming(
            source, base, out, _col_weights(), device="cpu",
            allow_unstamped_research=True,
            per_expert_config_path=per_config,
        )
    legacy_files = {
        path.relative_to(tmp_path / "legacy"): path.read_bytes()
        for path in (tmp_path / "legacy").rglob("*") if path.is_file()
    }
    uniform_files = {
        path.relative_to(tmp_path / "uniform"): path.read_bytes()
        for path in (tmp_path / "uniform").rglob("*") if path.is_file()
    }
    legacy_card_bytes = legacy_files.pop(Path("shipcard.json"))
    uniform_card_bytes = uniform_files.pop(Path("shipcard.json"))
    assert uniform_files == legacy_files
    assert len(uniform_card_bytes) == len(legacy_card_bytes)

    legacy_card = load_shipcard(tmp_path / "legacy" / "shipcard.json")
    uniform_card = load_shipcard(tmp_path / "uniform" / "shipcard.json")
    for key in (
        "model_sha",
        "artifact_bytes",
        "reserved_file_bytes",
        "build",
        "slots",
    ):
        assert uniform_card[key] == legacy_card[key]
    assert legacy_card["model_dir"] == str(tmp_path / "legacy")
    assert uniform_card["model_dir"] == str(tmp_path / "uniform")
    assert "per_expert_format_groups" not in json.loads(
        (tmp_path / "uniform" / "quant_config.json").read_text()
    )
