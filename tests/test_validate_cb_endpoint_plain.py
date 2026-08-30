"""The third CB decode topology: a plain artifact with no DSpark shape.

Every test here is a way the receipt should have come out differently. A cover
proof whose numbers cannot move is not evidence, which is the failure this mode
was written to avoid: before 2026-08-15 the completeness check enumerated only
FP8 ``.weight`` tensors, so a 27B NVFP4-CB export with 818 units was declared
complete on the strength of one of them.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import validate_cb_endpoint as cbv
from prismaquant.cb_layout import codebook_subtable_shapes, parse_format_name


_FP8_CB = "FP8_CB_K28"
_NVFP4_CB = "NVFP4_CB_K12"
_CB_UNITS = {
    "model.layers.0.mlp.gate_proj": _FP8_CB,
    "model.layers.0.mlp.up_proj": _FP8_CB,
    "model.layers.0.self_attn.q_proj": _NVFP4_CB,
}
_HEAD = "lm_head"


def _refs(format_name: str) -> list[str]:
    family, k = parse_format_name(format_name)
    shapes = codebook_subtable_shapes(k, family.mode, family.n_sub)
    if len(shapes) == 1:
        return [f"cb_codebook.lattice.{format_name}"]
    return [
        f"cb_codebook.lattice.{format_name}.sub{index}"
        for index in range(len(shapes))
    ]


def _codebook_tensors() -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for index, format_name in enumerate((_FP8_CB, _NVFP4_CB)):
        family, k = parse_format_name(format_name)
        shapes = codebook_subtable_shapes(k, family.mode, family.n_sub)
        for ref, shape in zip(_refs(format_name), shapes, strict=True):
            tensors[ref] = torch.full(
                tuple(shape), float(index + 1), dtype=torch.float16
            )
    return tensors


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.to(torch.float16).cpu().numpy().tobytes()
    ).hexdigest()


def _model_tensors() -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for unit, format_name in _CB_UNITS.items():
        tensors[f"{unit}.cb_qweight"] = torch.zeros((8, 16), dtype=torch.uint8)
        companion = (
            "input_global_scale"
            if format_name.startswith("NVFP4")
            else "weight_scale"
        )
        tensors[f"{unit}.{companion}"] = torch.ones((8, 1), dtype=torch.float32)
    tensors[f"{_HEAD}.weight"] = torch.zeros(
        (16, 8), dtype=torch.float8_e4m3fn
    )
    tensors[f"{_HEAD}.weight_scale"] = torch.ones((16, 1), dtype=torch.float32)
    # One plain float tensor, so `tensor_count` is not just the quantized ones.
    tensors["model.norm.weight"] = torch.ones((8,), dtype=torch.bfloat16)
    return tensors


def _quant_config(codebooks: dict[str, torch.Tensor]) -> dict:
    groups: dict[str, dict] = {}
    for index, format_name in enumerate((_FP8_CB, _NVFP4_CB)):
        family, k = parse_format_name(format_name)
        groups[f"group_{index}"] = {
            "format": format_name,
            "targets": sorted(
                unit for unit, name in _CB_UNITS.items() if name == format_name
            ),
            "scheme": {
                "codebook_ref": _refs(format_name),
                "codebook_source": "lattice",
                "k": k,
                "mode": family.mode,
                "n_sub": family.n_sub,
            },
        }
    groups["group_2"] = {
        "format": "float-quantized",
        "targets": [_HEAD],
        "weights": {
            "num_bits": 8,
            "strategy": "channel",
            "symmetric": True,
            "type": "float",
        },
    }
    return {
        "format": "nvfp4_cb",
        "quant_method": "gridbook",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": groups,
        "ignore": ["model.norm"],
        "provenance": {
            "renderer_abi": "prismaquant.nvfp4_cb_renderer.v1",
            "codebook_sha256": {
                ref: _digest(tensor) for ref, tensor in codebooks.items()
            },
        },
    }


def _write(root: Path, *, quant_config: dict, model_type: str = "qwen3") -> None:
    (root / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": ["Qwen3ForCausalLM"]}),
        encoding="utf-8",
    )
    (root / "quant_config.json").write_text(
        json.dumps(quant_config), encoding="utf-8"
    )


@pytest.fixture
def plain_case(tmp_path: Path) -> tuple[Path, dict]:
    codebooks = _codebook_tensors()
    save_file(_model_tensors(), str(tmp_path / "model.safetensors"))
    save_file(codebooks, str(tmp_path / "cb_codebooks.pqcb"))
    quant_config = _quant_config(codebooks)
    _write(tmp_path, quant_config=quant_config)
    return tmp_path, quant_config


def _validate(case: tuple[Path, dict]) -> dict:
    root, quant_config = case
    return cbv.validate_cb_artifact_decode_contract(root, quant_config)


def _rewrite(root: Path, quant_config: dict) -> dict:
    (root / "quant_config.json").write_text(
        json.dumps(quant_config), encoding="utf-8"
    )
    return quant_config


#: The plain receipt this fixture produced BEFORE the per-role/passthrough cover
#: work, captured from the pre-change code. See
#: `test_the_plain_receipt_digest_is_stable_for_an_ordinary_artifact`.
#:
#: The COVER digest has never moved and must not: it is computed over the
#: artifact's own bytes and geometry, which the Gridbook pin does not touch.
#: The EVIDENCE digest moved once, on the 0.8.11/v4 -> 0.9.1/v12 pin advance,
#: because the evidence envelope carries `required_runtime_contract_schema`
#: and the cover does not.  That was verified rather than assumed: substituting
#: `gridbook.runtime-contract.v4` back into
#: `validate_cb_endpoint.DSPARK_CB_RUNTIME_CONTRACT_SCHEMA` and rerunning this
#: test reproduces the previous evidence digest
#: e604772c53b1e3cba9783c0f961a62e5d0d89f45d077df69da4a89f8df50f223 exactly,
#: so the schema string is the ONLY input that changed.  A receipt that names
#: the contract it was validated against SHOULD move when that contract does;
#: what would be a defect is the cover moving, and it did not.
_PINNED_PLAIN_COVER_SHA256 = (
    "f4127eb51b5b586852ffb27b30de8cf90dda67608bf020480b77cfb285a8f4ac"
)
_PINNED_PLAIN_EVIDENCE_SHA256 = (
    "7426aafbe191ef9ecf78fcf6dd40fd164d0ffc31c367c37f12cb558a2417f41d"
)


def test_a_plain_cb_artifact_proves_its_own_cover(plain_case) -> None:
    evidence = _validate(plain_case)

    assert evidence["schema"] == cbv.ARTIFACT_DECODE_CONTRACT_SCHEMA_PLAIN
    assert evidence["mode"] == cbv.CB_PLAIN_MODE
    assert evidence["complete"] is True
    # Three CB units and the FP8 head, all four claimed, over six tensors more
    # than that: the count could have been smaller and was not.
    assert evidence["cb_unit_count"] == len(_CB_UNITS) + 1
    assert evidence["quantized_unit_count"] == len(_CB_UNITS) + 1
    assert evidence["tensor_count"] > evidence["quantized_unit_count"]
    assert evidence["codebook_ref_count"] == len(_refs(_FP8_CB)) + len(
        _refs(_NVFP4_CB)
    )
    assert evidence["required_runtime_features"] == {}
    assert {entry["group"] for entry in evidence["group_cover"]} == {
        "group_0",
        "group_1",
        "group_2",
    }
    for entry in evidence["group_cover"]:
        assert entry["target_count"] == entry["unit_count"]
        assert entry["planes"]


def test_a_target_dropped_from_a_group_leaves_its_tensor_unclaimed(
    plain_case,
) -> None:
    root, quant_config = plain_case
    mutated = copy.deepcopy(quant_config)
    mutated["config_groups"]["group_0"]["targets"] = [
        "model.layers.0.mlp.gate_proj"
    ]
    _rewrite(root, mutated)

    with pytest.raises(Exception) as excinfo:
        cbv.validate_cb_artifact_decode_contract(root, mutated)
    assert "model.layers.0.mlp.up_proj" in str(excinfo.value)


def test_a_group_whose_units_disagree_on_planes_is_half_exported(
    plain_case,
) -> None:
    root, quant_config = plain_case
    tensors = _model_tensors()
    del tensors["model.layers.0.mlp.up_proj.weight_scale"]
    save_file(tensors, str(root / "model.safetensors"))

    with pytest.raises(cbv.CBEndpointValidationError, match="same planes"):
        cbv.validate_cb_artifact_decode_contract(root, quant_config)


def test_a_codebook_ref_the_pqcb_lacks_is_refused(plain_case) -> None:
    root, quant_config = plain_case
    mutated = copy.deepcopy(quant_config)
    refs = mutated["config_groups"]["group_0"]["scheme"]["codebook_ref"]
    refs[0] = refs[0].replace("lattice", "learned")
    digests = mutated["provenance"]["codebook_sha256"]
    digests[refs[0]] = digests.pop(
        quant_config["config_groups"]["group_0"]["scheme"]["codebook_ref"][0]
    )
    _rewrite(root, mutated)

    with pytest.raises(
        cbv.CBEndpointValidationError, match="do not exactly equal"
    ):
        cbv.validate_cb_artifact_decode_contract(root, mutated)


def test_a_codebook_payload_mutation_is_refused(plain_case) -> None:
    root, quant_config = plain_case
    codebooks = _codebook_tensors()
    ref = _refs(_FP8_CB)[0]
    codebooks[ref][0, 0] += torch.tensor(1, dtype=torch.float16)
    save_file(codebooks, str(root / "cb_codebooks.pqcb"))

    with pytest.raises(cbv.CBEndpointValidationError, match="payload SHA-256"):
        cbv.validate_cb_artifact_decode_contract(root, quant_config)


def test_a_quantized_tensor_listed_in_ignore_is_a_contradiction(
    plain_case,
) -> None:
    root, quant_config = plain_case
    mutated = copy.deepcopy(quant_config)
    mutated["ignore"].append("model.layers.0.mlp.gate_proj")
    _rewrite(root, mutated)

    with pytest.raises(Exception, match="ignore"):
        cbv.validate_cb_artifact_decode_contract(root, mutated)


def test_a_dsv4_lane_artifact_cannot_present_as_plain(plain_case) -> None:
    """Losing the bridge is still a refusal.

    A DSv4 body that ships without MTP may present as plain, but only by
    RECORDING the omission (see the split-topology tests). This fixture records
    nothing, which is what an artifact that merely lost its overlay looks like,
    so the lane guard still fails closed on it.
    """

    root, quant_config = plain_case
    _write(root, quant_config=quant_config, model_type="deepseek_v4")

    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="one of dspark_source_overlay or dspark_cb_sidecar",
    ):
        cbv.validate_cb_artifact_decode_contract(root, quant_config)


def test_the_plain_receipt_digest_is_stable_for_an_ordinary_artifact(
    plain_case,
) -> None:
    """Pinned so a cover change cannot silently move every artifact's receipt.

    Per-role routed claims and source passthrough both changed how the cover is
    computed (2026-08-16). Neither may move the numbers for an artifact that has
    neither, which is every plain CB artifact validated before that date. If a
    future change to the cover legitimately alters this, recompute BOTH digests
    from the pre-change code and say in the commit why the receipt may move.

    The digests are recomputed from the fixture here rather than compared to
    themselves -- a receipt pin that cannot fail is not a pin.
    """

    evidence = _validate(plain_case)

    assert evidence["cover_sha256"] == _PINNED_PLAIN_COVER_SHA256
    assert evidence["evidence_sha256"] == _PINNED_PLAIN_EVIDENCE_SHA256


def test_a_dspark_topology_off_the_dsv4_lane_is_refused(plain_case) -> None:
    root, quant_config = plain_case
    mutated = copy.deepcopy(quant_config)
    mutated["provenance"]["dspark_cb_sidecar"] = {"schema": "whatever"}
    _rewrite(root, mutated)

    with pytest.raises(
        cbv.CBEndpointValidationError, match="only the 'dsv4_flash' lane"
    ):
        cbv.validate_cb_artifact_decode_contract(root, mutated)


@pytest.mark.parametrize(
    "field, value",
    [
        ("cb_unit_count", 99),
        ("quantized_unit_count", 99),
        ("codebook_ref_count", 0),
    ],
)
def test_a_receipt_edited_after_the_fact_is_stale(
    plain_case, field: str, value: int
) -> None:
    evidence = dict(_validate(plain_case))
    evidence[field] = value

    with pytest.raises(cbv.CBEndpointValidationError):
        cbv._validate_artifact_decode_record(evidence)


def test_a_receipt_cannot_borrow_the_dspark_schema(plain_case) -> None:
    evidence = dict(_validate(plain_case))
    evidence["schema"] = cbv.ARTIFACT_DECODE_CONTRACT_SCHEMA_V2

    with pytest.raises(cbv.CBEndpointValidationError):
        cbv._validate_artifact_decode_record(evidence)


def test_the_runtime_pin_must_implement_every_feature_the_artifact_needs(
    plain_case,
) -> None:

    evidence = dict(_validate(plain_case))
    evidence["required_runtime_features"] = {
        cbv.CB_ROUTED_MOE_RUNTIME_FEATURE: cbv.CB_ROUTED_MOE_RUNTIME_FEATURE_VERSION
    }

    with pytest.raises(cbv.CBEndpointValidationError, match="decode ABI"):
        cbv._require_artifact_decode_runtime_features(
            evidence,
            {
                "runtime_contract_schema": cbv.DSPARK_CB_RUNTIME_CONTRACT_SCHEMA,
                "required_abi_features": {},
            },
        )
