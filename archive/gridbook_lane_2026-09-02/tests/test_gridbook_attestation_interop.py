"""One process runs BOTH halves of the K0.2 stage-attestation contract.

Everything else that guards the routed-MoE stage attestation is two suites
that never execute each other's code: ``tests/test_nvfp4_activation_contract``
exercises this emitter, Gridbook's same-named suite hand-writes a stage
section that *reimplements* it, and the two are joined only by pinned digest
hexes and schema literals.  Those pins catch a changed value.  They cannot
catch a changed *shape*, and the shape has a trap in it: Gridbook's parser
requires every stage entry to declare EXACTLY its five attested fields, extra
keys rejected, while every digest in the contract is framed over those same
five fields by name.  An "additive, backwards-compatible" field added to a
stage entry on the producer side therefore moves no hex anywhere, leaves both
suites green, and fails for the first time when vLLM loads the artifact.

This file is the only place a routed-MoE record emitted by the real producer
is fed to the real pinned consumer -- ``parse_contract``,
``validate_payload``, ``verify_routed_moe_stages`` and the artifact-level K0.2
verdict -- so that failure lands in CI with the producer's own field names in
the message.  Like the other cross-repository tests here, PrismaQuant never
imports Gridbook outside this dedicated job.
"""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.nvfp4_activation_contract import (
    CALIBRATION_SOURCE_SUPPLEMENTAL_MODULE_INPUT,
    CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
    CALIBRATION_SOURCE_TARGET_CACHE,
    FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
    NVFP4_ACTIVATION_CONTRACT_KEY,
    NVFP4_ACTIVATION_CONTRACT_SCHEMA,
    NVFP4_ACTIVATION_CONTRACT_SCHEMA_V2,
    NVFP4_INPUT_GLOBAL_SCALE_SUFFIX,
    NVFP4_ROUTED_MOE_STAGE_KEY,
    NVFP4_ROUTED_MOE_STAGES,
    build_execution_contract,
    calibrated_input_global_scales_with_sources,
    input_global_scale_tensor,
    routed_moe_stages_sha256,
    stage_values_sha256,
    target_values_sha256,
)


REQUIRE_CONTRACT = os.environ.get(
    "PRISMAQUANT_REQUIRE_GRIDBOOK_CONTRACT") == "1"

pytestmark = pytest.mark.skipif(
    not REQUIRE_CONTRACT,
    reason="run by the pinned Gridbook compatibility CI job",
)

_MODULE = "model.layers.0.mlp.experts"
_W13 = f"{_MODULE}.gate_up_proj"
_W2 = f"{_MODULE}.down_proj"
_DENSE = "model.layers.0.self_attn.q_proj"
_POLICY = FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY

# The five fields a stage entry attests.  This list is not decoration: it is
# simultaneously what the producer emits, what every stage digest is framed
# over, and -- the trap -- the exact set Gridbook's parser demands, no more.
_ATTESTED_STAGE_FIELDS = [
    "calibration_source",
    "input_global_scale_policy",
    "stage",
    "stage_values_sha256",
    "target",
]


@pytest.fixture(scope="module")
def gridbook_contract():
    """The pinned external runtime's parser half, or a clean skip."""

    pytest.importorskip("gridbook")
    return pytest.importorskip("gridbook.nvfp4_activation_contract")


@pytest.fixture(scope="module")
def gridbook_validation():
    """The pinned runtime's artifact-level K0.2 verdict, or a clean skip."""

    pytest.importorskip("gridbook")
    return pytest.importorskip("gridbook._fused_nvfp4_validation")


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


def _write_activation(
    cache_dir: Path, name: str, inputs: torch.Tensor
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
    torch.save({"name": name, "inputs": inputs}, cache_dir / filename)


def _routed_moe_checkpoint(root: Path) -> tuple[Path, Path]:
    """Minimal per-expert MoE checkpoint plus the probe cache entries.

    Fixture conventions follow ``tests/test_nvfp4_activation_contract.py``:
    two experts and split gate/up leaves on disk, with the experts-module
    input cached but no routed intermediate, so ``w2`` can only be calibrated
    from the replay.  One dense attention entry rides along so the emitted
    record spans both a stage target and an ordinary Linear.
    """

    hidden, inter, experts = 16, 8, 2
    model_dir = root / "moe"
    act_dir = root / "moe_act"
    model_dir.mkdir(parents=True)
    generator = torch.Generator().manual_seed(20260801)
    tensors = {
        "model.layers.0.mlp.gate.weight": torch.randn(
            experts, hidden, generator=generator
        ),
    }
    for expert in range(experts):
        for leaf in ("gate_proj", "up_proj"):
            tensors[f"{_MODULE}.{expert}.{leaf}.weight"] = torch.randn(
                inter, hidden, generator=generator
            )
    save_file(tensors, str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(json.dumps(
        {"num_experts_per_tok": 1, "norm_topk_prob": True}
    ))
    _write_activation(
        act_dir, _MODULE, torch.randn(32, hidden, generator=generator)
    )
    _write_activation(
        act_dir, _DENSE, torch.randn(32, hidden, generator=generator)
    )
    return model_dir, act_dir


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    """Emit a routed-MoE and a dense-only record through the real emitter.

    Nothing here is hand-written: the scales and the calibration sources both
    come out of the production calibration path, so the record under test is
    the one an export would actually ship.
    """

    from prismaquant.moe_imatrix import (
        synthesize_packed_expert_activation_samples,
    )

    root = tmp_path_factory.mktemp("gridbook-attestation-interop")
    model_dir, act_dir = _routed_moe_checkpoint(root)
    profile = _PackedExpertProfile()
    supplemental = synthesize_packed_expert_activation_samples(
        model_dir, act_dir, {_W13, _W2}, profile, device="cpu",
    )
    scales, sources = calibrated_input_global_scales_with_sources(
        [_W13, _W2, _DENSE],
        activation_cache_dir=act_dir,
        policy=_POLICY,
        profile=profile,
        supplemental_activations=supplemental,
        calibration_device="cpu",
    )
    # The two stages must not have collapsed onto one calibrated tensor, or
    # the record under test would not exercise the stage distinction at all.
    assert sources == {
        _W13: CALIBRATION_SOURCE_SUPPLEMENTAL_MODULE_INPUT,
        _W2: CALIBRATION_SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
        _DENSE: CALIBRATION_SOURCE_TARGET_CACHE,
    }
    assert scales[_W13] != scales[_W2]

    routed_record, routed_scales = build_execution_contract(
        scales, policy=_POLICY, calibration_sources=sources, profile=profile,
    )
    dense_record, dense_scales = build_execution_contract(
        {_DENSE: scales[_DENSE]},
        policy=_POLICY,
        calibration_sources={_DENSE: sources[_DENSE]},
        profile=profile,
    )
    return {
        "routed": (routed_record, routed_scales),
        "dense": (dense_record, dense_scales),
    }


def _write_artifact(root: Path, record, scales) -> Path:
    """Write the artifact shape Gridbook's K0.2 verdict actually reads.

    ``execution_contracts.<key>`` is where
    ``prismaquant.cb_export_config.build_quant_config`` puts the record, and
    ``<target>.input_global_scale`` F32[1] is the producer's own serialized
    scalar; the verdict needs no GPU, no vLLM, and no engine to read them.
    """

    root.mkdir(parents=True, exist_ok=True)
    (root / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "config_groups": {},
        "execution_contracts": {NVFP4_ACTIVATION_CONTRACT_KEY: record},
    }, indent=2, sort_keys=True), encoding="utf-8")
    save_file(
        {
            f"{target}.{NVFP4_INPUT_GLOBAL_SCALE_SUFFIX}":
                input_global_scale_tensor(value)
            for target, value in scales.items()
        },
        str(root / "model.safetensors"),
    )
    return root


def test_emitted_routed_moe_record_parses_and_verifies_in_gridbook(
    emitted, gridbook_contract
):
    contract = gridbook_contract
    record, scales = emitted["routed"]

    parsed = contract.parse_contract(
        {"execution_contracts": {NVFP4_ACTIVATION_CONTRACT_KEY: record}}
    )
    assert parsed is not None
    assert parsed["schema"] == contract.CONTRACT_SCHEMA_V2
    assert parsed["schema"] == NVFP4_ACTIVATION_CONTRACT_SCHEMA_V2

    stages = contract.parse_routed_moe_stages(record)
    assert sorted(stages) == [_MODULE]
    assert list(stages[_MODULE]) == list(contract.ROUTED_MOE_STAGES)
    assert stages[_MODULE]["w13"]["target"] == _W13
    assert stages[_MODULE]["w2"]["target"] == _W2
    # w2 was calibrated on the routed intermediate, not the module input --
    # the whole reason the stage section exists.
    assert stages[_MODULE]["w2"]["calibration_source"] == (
        contract.SOURCE_SUPPLEMENTAL_ROUTED_REPLAY
    )
    # The producer emits exactly the field set the consumer's strict parser
    # allows.  See the additive-field test for why "exactly" is load-bearing.
    for stage in NVFP4_ROUTED_MOE_STAGES:
        assert sorted(stages[_MODULE][stage]) == _ATTESTED_STAGE_FIELDS

    assert contract.validate_payload(record, scales) == scales

    verified = contract.verify_routed_moe_stages(record, scales)
    assert verified["verdict"] == "attested_and_verified"
    assert verified["attested"] is True
    assert verified["modules"] == [_MODULE]
    assert verified["failing_module"] is None
    assert verified["failing_stage"] is None


def test_emitted_routed_moe_artifact_earns_the_k02_attested_verdict(
    emitted, gridbook_contract, gridbook_validation, tmp_path
):
    common = gridbook_validation
    record, scales = emitted["routed"]
    artifact = _write_artifact(tmp_path / "routed", record, scales)

    # The record survives the JSON round trip the artifact imposes on it.
    read_record, read_scales = common.read_artifact_activation_contract(
        artifact
    )
    assert read_record == record
    assert read_scales == scales

    verdict = common.k02_readiness_verdict(artifact, mode="moe128")
    assert verdict["verdict"] == common.K02_ATTESTED
    assert verdict["pass"] is True
    assert verdict["evidence_class"] == common.EVIDENCE_STAGE_ATTESTED
    assert verdict["fused_moe_ab_is_evidence"] is True
    assert verdict["modules"] == [_MODULE]
    assert verdict["contract_schema"] == NVFP4_ACTIVATION_CONTRACT_SCHEMA_V2


def test_emitted_dense_only_record_stays_gridbook_v1_valid(
    emitted, gridbook_contract, gridbook_validation, tmp_path
):
    contract = gridbook_contract
    common = gridbook_validation
    record, scales = emitted["dense"]

    assert record["schema"] == NVFP4_ACTIVATION_CONTRACT_SCHEMA
    assert NVFP4_ROUTED_MOE_STAGE_KEY not in record
    parsed = contract.parse_contract(
        {"execution_contracts": {NVFP4_ACTIVATION_CONTRACT_KEY: record}}
    )
    assert parsed is not None
    assert parsed["schema"] == contract.CONTRACT_SCHEMA
    assert contract.parse_routed_moe_stages(record) is None
    assert contract.validate_payload(record, scales) == scales

    # A dense-only artifact carries no stage claim; that is valid for a dense
    # run and correctly disqualifying as fused-MoE evidence.
    unattested = contract.verify_routed_moe_stages(record, scales)
    assert unattested["verdict"] == "not_attested"
    assert unattested["attested"] is False

    artifact = _write_artifact(tmp_path / "dense", record, scales)
    dense = common.k02_readiness_verdict(artifact, mode="dense")
    assert dense["verdict"] == common.K02_NOT_ATTESTED
    assert dense["pass"] is True
    assert dense["evidence_class"] == common.EVIDENCE_DENSE_SCOPE
    routed = common.k02_readiness_verdict(artifact, mode="moe128")
    assert routed["pass"] is False
    assert routed["evidence_class"] == common.EVIDENCE_FALLBACK_TELEMETRY


def test_additive_stage_entry_field_is_digest_invisible_but_gridbook_rejects(
    emitted, gridbook_contract, gridbook_validation, tmp_path
):
    """The trap: an additive producer field both suites would call harmless."""

    contract = gridbook_contract
    common = gridbook_validation
    record, scales = emitted["routed"]

    mutated = copy.deepcopy(record)
    section = mutated[NVFP4_ROUTED_MOE_STAGE_KEY]
    entry = section["modules"][_MODULE]["w13"]
    assert sorted(entry) == _ATTESTED_STAGE_FIELDS
    # Exactly the shape of a field a producer would add believing it additive.
    entry["stage_input_dtype"] = "bfloat16"

    # 1. No digest on either side can see it.  Every digest in the contract is
    #    framed over the five attested fields by name, so the extra key leaves
    #    the per-stage hex, the section hex, and the whole-model hex identical
    #    -- which is why the pinned-hex tests in both repositories stay green.
    assert stage_values_sha256(
        stage="w13",
        target=entry["target"],
        policy=_POLICY,
        calibration_source=entry["calibration_source"],
        value=scales[_W13],
    ) == entry["stage_values_sha256"]
    assert routed_moe_stages_sha256(section["modules"]) == (
        section["stages_sha256"]
    )
    assert contract.routed_moe_stages_sha256(section["modules"]) == (
        section["stages_sha256"]
    )
    assert target_values_sha256(scales, policy=_POLICY) == (
        mutated["target_values_sha256"]
    )

    # 2. The consumer rejects the artifact anyway: the stage entry field set
    #    is a strict equality, so this is a hard load-time failure that no
    #    digest, schema literal, or producer-side test would have caught.
    expected = (
        f"routed-MoE module {_MODULE!r} stage w13 must declare exactly "
        f"{_ATTESTED_STAGE_FIELDS}"
    )
    with pytest.raises(ValueError) as parse_error:
        contract.parse_routed_moe_stages(mutated)
    assert str(parse_error.value) == expected
    with pytest.raises(ValueError) as record_error:
        contract.parse_contract(
            {"execution_contracts": {NVFP4_ACTIVATION_CONTRACT_KEY: mutated}}
        )
    assert str(record_error.value) == expected

    malformed = contract.verify_routed_moe_stages(mutated, scales)
    assert malformed["verdict"] == "malformed_stage_attestation"
    assert malformed["detail"] == expected

    # 3. End to end, the artifact an operator would ship reads back unusable.
    artifact = _write_artifact(tmp_path / "additive", mutated, scales)
    verdict = common.k02_readiness_verdict(artifact, mode="moe128")
    assert verdict["verdict"] == common.K02_ARTIFACT_UNREADABLE
    assert verdict["detail"] == f"ValueError: {expected}"
    assert verdict["pass"] is False


def test_producer_and_runtime_digest_implementations_agree(
    emitted, gridbook_contract
):
    """Both digest implementations, run here -- not two pinned hexes."""

    contract = gridbook_contract
    record, scales = emitted["routed"]

    assert target_values_sha256(scales, policy=_POLICY) == (
        contract.target_values_sha256(scales, policy=_POLICY)
    ) == record["target_values_sha256"]

    section = record[NVFP4_ROUTED_MOE_STAGE_KEY]
    modules = section["modules"]
    assert list(NVFP4_ROUTED_MOE_STAGES) == list(contract.ROUTED_MOE_STAGES)
    for module in sorted(modules):
        for stage in NVFP4_ROUTED_MOE_STAGES:
            entry = modules[module][stage]
            fields = {
                "stage": stage,
                "target": entry["target"],
                "policy": _POLICY,
                "calibration_source": entry["calibration_source"],
            }
            value = scales[entry["target"]]
            assert stage_values_sha256(**fields, value=value) == (
                contract.stage_values_sha256(**fields, value=value)
            ) == entry["stage_values_sha256"]

    assert routed_moe_stages_sha256(modules) == (
        contract.routed_moe_stages_sha256(modules)
    ) == section["stages_sha256"]
