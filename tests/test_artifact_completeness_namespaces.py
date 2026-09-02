"""The completeness gate must read a config-group target in the namespace the
exporter actually wrote it in.

THE BUG THIS PINS. A DELEGATED config group is spelled in vLLM's module
namespace, because compressed-tensors matches its targets against vLLM's module
tree at load: `export_nvfp4_cb_streaming._delegated_target_name` is
`profile.to_vllm_internal_name`. On every architecture this gate had seen, that
map was the identity, so the difference never showed. It is not the identity on
a multimodal wrapper — Qwen3.8-27B stores `lm_head.weight` while vLLM builds the
head at `language_model.lm_head` — and the gate rejected a *correct* 12.98 GB
artifact at the end of a 50-minute export with "1 scale-bearing weight(s) are
claimed by no mechanism at all: ['lm_head']".

The fix maps the UNIT forward through the profile rather than inverting the
claim, because `to_vllm_internal_name` is the producer's own map and has no
inverse to call. These tests exercise that seam directly with a stub profile:
the real profile needs a full checkpoint to detect, and the property under test
is about the namespaces, not about any one architecture.
"""
from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from prismaquant.artifact_completeness import (
    ArtifactIncomplete,
    assert_artifact_complete,
    check_artifact_completeness,
)
import prismaquant.artifact_completeness as completeness


class _WrapperProfile:
    """The two maps a multimodal wrapper profile provides, and nothing else.

    `lm_head` keeps its checkpoint spelling but moves under `language_model` in
    vLLM's tree; the body keeps `model.` on disk and gains the wrapper prefix in
    vLLM. Both are taken from the real Qwen3_5DenseProfile's answers.
    """

    def to_vllm_internal_name(self, name: str) -> str:
        if name == "lm_head":
            return "language_model.lm_head"
        if name.startswith("model.language_model."):
            return "language_model.model." + name[len("model.language_model."):]
        return name

    def source_tensor_name(self, name: str) -> str:
        return name


def _write_artifact(root: Path, *, targets, tensors) -> None:
    """A minimal artifact: one safetensors shard header plus a quant_config."""

    root.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {}
    offset = 0
    for name, dtype, shape, span in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + span],
        }
        offset += span
    blob = json.dumps(header).encode("utf-8")
    with (root / "model.safetensors").open("wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(b"\0" * offset)
    (root / "quant_config.json").write_text(json.dumps({
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": list(targets),
                "weights": {"num_bits": 8, "type": "float",
                            "strategy": "channel"},
                "input_activations": None,
            },
        },
        "ignore": [],
    }), encoding="utf-8")


_LM_HEAD = (
    ("lm_head.weight", "F8_E4M3", (32, 8), 32 * 8),
    ("lm_head.weight_scale", "F32", (32, 1), 32 * 4),
)


@pytest.fixture()
def wrapper_profile(monkeypatch):
    monkeypatch.setattr(
        completeness, "_detect_profile_quietly",
        lambda _root: _WrapperProfile())


def test_vllm_spelled_group_target_claims_its_checkpoint_unit(
        tmp_path, wrapper_profile):
    """The exact shape of the Qwen3.8-27B artifact-A failure."""

    root = tmp_path / "artifact"
    _write_artifact(
        root,
        targets=["re:^language_model[.]lm_head$"],
        tensors=_LM_HEAD,
    )
    report = check_artifact_completeness(root)
    assert report.undeclared == []
    assert report.orphan_scale == []
    assert report.cb_units == ["lm_head"]
    assert report.ok
    assert_artifact_complete(root)


def test_checkpoint_spelled_group_target_still_claims_its_unit(
        tmp_path, wrapper_profile):
    """The forward map is additive: an identity-namespace artifact, which is
    every pre-wrapper artifact ever shipped, keeps passing unchanged."""

    root = tmp_path / "artifact"
    _write_artifact(root, targets=["re:^lm_head$"], tensors=_LM_HEAD)
    report = check_artifact_completeness(root)
    assert report.cb_units == ["lm_head"]
    assert report.ok


def test_a_group_for_a_different_module_still_fails(tmp_path, wrapper_profile):
    """The negative control. Mapping the unit forward must not turn the gate
    into one that accepts any target at all: a group naming a NEIGHBOURING
    module leaves the head unclaimed, which is the bug the gate exists for."""

    root = tmp_path / "artifact"
    _write_artifact(
        root,
        targets=["re:^language_model[.]model[.]layers[.]0[.]mlp[.]down_proj$"],
        tensors=_LM_HEAD,
    )
    with pytest.raises(ArtifactIncomplete, match="claimed by no mechanism"):
        assert_artifact_complete(root)


def test_without_a_profile_the_gate_falls_back_to_the_literal_name(tmp_path,
                                                                  monkeypatch):
    """`_detect_profile_quietly` returns None on an architecture this build
    does not know. The gate must still run — and on a vLLM-spelled target it
    then has no way to resolve the unit, so it reports rather than guessing."""

    monkeypatch.setattr(completeness, "_detect_profile_quietly",
                        lambda _root: None)
    root = tmp_path / "artifact"
    _write_artifact(
        root,
        targets=["re:^language_model[.]lm_head$"],
        tensors=_LM_HEAD,
    )
    with pytest.raises(ArtifactIncomplete, match="claimed by no mechanism"):
        assert_artifact_complete(root)


def test_split_format_group_fused_units_resolve_through_the_projection():
    """A per-expert split-format unit spells its group token AFTER the
    projection, so the fusion map has to be applied to the projection and the
    token re-attached.

    Reading `format_group_fp8_cb_k28` as the leaf finds nothing in
    `packed_modules_mapping`, and a split export — whose config groups MUST
    name the unfused halves, since vLLM canonical scheme names are a hard
    serving invariant — then reports every fused stack it ships as unclaimed.
    """

    fused = {"gate_up_proj": ("gate_proj", "up_proj")}
    stack = "model.layers.0.mlp.experts.gate_up_proj"

    assert completeness._fused_member_units(stack, fused) == (
        "model.layers.0.mlp.experts.gate_proj",
        "model.layers.0.mlp.experts.up_proj",
    )
    assert completeness._fused_member_units(
        f"{stack}.format_group_fp8_cb_k28", fused
    ) == (
        "model.layers.0.mlp.experts.gate_proj.format_group_fp8_cb_k28",
        "model.layers.0.mlp.experts.up_proj.format_group_fp8_cb_k28",
    )
    # An UNFUSED projection carrying the same token stays unmapped: the bridge
    # must not invent members for a unit the fusion map says nothing about.
    assert completeness._fused_member_units(
        "model.layers.0.mlp.experts.down_proj.format_group_fp8_cb_k28", fused
    ) == ()


# `test_a_sidecar_alias_map_is_empty_without_a_published_sidecar` was deleted
# on 2026-09-02. It pinned `completeness._dspark_sidecar_aliases`, the opt-in
# resolver for the DSpark CB sidecar's physical->construction alias map. The
# sidecar was a Gridbook codebook-lane artifact and the resolver went to
# archive/gridbook_lane_2026-09-02/ with the lane; nothing publishes a
# `dspark_cb_sidecar` record any more.


# --- THE FIFTH NAMESPACE: DSpark physical vs construction — REMOVED --------
#
# Deleted on 2026-09-02 with the Gridbook codebook lane
# (archive/gridbook_lane_2026-09-02/). The section covered the DSpark CB
# sidecar: a draft ships its tensors as `mtp.{stage}.<tail>` while vLLM builds
# those blocks as body layers past the end of the body, so the CB exporter
# wrote their config-group targets as
# `model.layers.{num_hidden_layers+stage}.<tail>`
# (`_cb_target_name` -> `dspark_cb_construction_target_for_physical_output`).
# The three tests here proved the completeness bridge resolved that spelling
# on a declared sidecar, refused it for the wrong stage, and stayed inert
# everywhere else.
#
# The exporter that wrote those targets, the resolver that read them, and the
# `dspark_cb_sidecar` provenance record are all archived; the `mtp.dspark`
# shipcard slot went with them. Nothing produces a DSpark CB sidecar, so the
# bridge has no artifact to bridge. The other four namespaces in this file are
# unaffected and still run.


@pytest.fixture()
def no_profile(monkeypatch):
    """The bridge is architecture arithmetic, not a profile map: it must work
    with no profile at all, which is also what a synthetic artifact detects."""

    monkeypatch.setattr(
        completeness, "_detect_profile_quietly", lambda _root: None)


# --- A SPLIT expert bank is claimed by its own declaration -----------------
#
# A mixed-rung expert stack ships one tensor per rung,
# `…gate_up_proj.format_group_<wire>`, claimed by `per_expert_format_groups`
# rather than by a config group. `_validate_per_expert_format_groups` owns
# those tensors in both directions, so the classifier must recognize the
# mechanism instead of reporting them a second time as claimed by nothing.

_SPLIT_PARENT = "model.layers.0.mlp.experts.gate_up_proj"
_SPLIT_UNITS = (
    (f"{_SPLIT_PARENT}.format_group_fp8_cb_k28.cb_qweight", "U8", (4, 8), 32),
    (f"{_SPLIT_PARENT}.format_group_fp8_cb_k29.cb_qweight", "U8", (4, 8), 32),
)


def _write_split_group_artifact(root: Path, *, declare: bool) -> None:
    _write_artifact(root, targets=[], tensors=_SPLIT_UNITS)
    if not declare:
        return
    quant = json.loads((root / "quant_config.json").read_text())
    quant["per_expert_format_groups"] = {
        "version": 1,
        "layers": {"0": {
            family: [
                {
                    "format_wire_id": wire,
                    "expert_ids": [0, 1],
                    "tensor_prefix": f"{_SPLIT_PARENT}.format_group_{wire}",
                }
                for wire in ("fp8_cb_k28", "fp8_cb_k29")
            ]
            for family in ("w13", "w2")
        }},
    }
    (root / "quant_config.json").write_text(
        json.dumps(quant), encoding="utf-8")


def test_a_declared_split_format_group_claims_its_tensor(
        tmp_path, no_profile):
    """The regression `1ccdf58` introduced: once the enumerator could see
    `.cb_qweight` planes it saw these too, and no branch knew the mechanism."""

    root = tmp_path / "artifact"
    _write_split_group_artifact(root, declare=True)
    report = check_artifact_completeness(root)
    assert report.undeclared == []
    assert sorted(report.cb_units) == [
        f"{_SPLIT_PARENT}.format_group_fp8_cb_k28",
        f"{_SPLIT_PARENT}.format_group_fp8_cb_k29",
    ]


def test_an_undeclared_split_format_group_still_fails(tmp_path, no_profile):
    """The negative control. Recognizing the declaration must not make a split
    tensor that NOTHING declares acceptable -- it stays a failure, reported by
    the per-expert validator rather than by the classifier."""

    root = tmp_path / "artifact"
    _write_split_group_artifact(root, declare=False)
    report = check_artifact_completeness(root)
    assert not report.ok
    assert sorted(report.undeclared_group_tensors) == [
        f"{_SPLIT_PARENT}.format_group_fp8_cb_k28.cb_qweight",
        f"{_SPLIT_PARENT}.format_group_fp8_cb_k29.cb_qweight",
    ]
    with pytest.raises(ArtifactIncomplete):
        assert_artifact_complete(root)
