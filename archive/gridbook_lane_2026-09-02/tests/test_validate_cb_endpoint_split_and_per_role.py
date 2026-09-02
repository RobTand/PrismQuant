"""Three DSv4 topologies, and a cover that can read a per-role routed stack.

TWO GAPS, BOTH FOUND ON ONE ARTIFACT (2026-08-16, the DSv4-Flash 92 GB body).

**The third topology.** The decode contract knew two shapes for a DSv4 release:
the in-band artifact that carries the MTP construction stages
(``dspark_source_overlay``) and the draft sidecar that IS the ``mtp.`` subset
(``dspark_cb_sidecar``). A 92 GB split release is neither: the body excludes
``mtp.`` because the draft ships beside it as a second artifact. The guard read
"declares neither" as "lost its bridge" and refused. It now allows the body to
declare itself by RECORDING the omission, which keeps the original fault --
an artifact that merely lost its overlay records nothing -- a refusal.

**The per-role cover.** ``_validate_plain_cb_artifact`` resolved config-group
targets to on-disk tensors one tensor at a time. A per-role learned CB layer
ships one packed ``…experts.gate_up_proj`` and names it with two groups, one per
half, because a per-role book fits one ``(layer, projection)`` and a packed
target binds exactly one ``codebook_ref``. Read per tensor, both groups look
empty and the tensor looks unclaimed. The cover is now computed per ROLE, which
is the pinned runtime's own rule (gridbook ``_resolve_moe_codebook_roles``, read
at 0.8.5 and re-read unchanged at the pinned 0.8.11:
collects a book per role, refuses two targets claiming one role, and refuses a
stack missing a role).

A third fix rides along: ``source_passthrough`` is a declared mechanism the
plain cover had never seen, because it had only ever run on artifacts with none.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import prismaquant.artifact_completeness as completeness
from prismaquant import validate_cb_endpoint as cbv
from prismaquant.cb_layout import codebook_subtable_shapes, parse_format_name


_FMT = "FP8_CB_K28"
_PACKED = "model.layers.0.mlp.experts.gate_up_proj"
_GATE = "model.layers.0.mlp.experts.gate_proj"
_UP = "model.layers.0.mlp.experts.up_proj"
_DENSE = "model.layers.0.self_attn.q_proj"
_PASSTHROUGH = "model.layers.0.self_attn.o_proj"


class _RoutedProfile:
    """Shaped like DeepseekV4Profile on the axes under test.

    No vLLM class, therefore no fused-sibling mapping; a declared packed-expert
    decomposition, therefore a routed w13 split. Identity name maps so the
    namespace bridges stay out of the way.
    """

    def fused_sibling_leaf_mapping(self):
        return {}

    def packed_expert_projection_names(self, param_name: str):
        if param_name == "gate_up_proj":
            return ("gate_proj", "up_proj")
        return (str(param_name),)

    def to_vllm_internal_name(self, name: str) -> str:
        return name

    def source_tensor_name(self, name: str) -> str:
        return name


@pytest.fixture()
def routed_profile(monkeypatch):
    monkeypatch.setattr(
        completeness, "_detect_profile_quietly", lambda _root: _RoutedProfile()
    )


def _role_refs(unit: str) -> list[str]:
    """A LEARNED book, named per (layer, projection) -- one unit, one book."""

    family, k = parse_format_name(_FMT)
    shapes = codebook_subtable_shapes(k, family.mode, family.n_sub)
    return [
        f"cb_codebook.{unit}.{_FMT}.sub{index}" for index in range(len(shapes))
    ]


def _codebooks(units) -> dict[str, torch.Tensor]:
    family, k = parse_format_name(_FMT)
    shapes = codebook_subtable_shapes(k, family.mode, family.n_sub)
    tensors: dict[str, torch.Tensor] = {}
    for offset, unit in enumerate(units):
        for ref, shape in zip(_role_refs(unit), shapes, strict=True):
            tensors[ref] = torch.full(
                tuple(shape), float(offset + 1), dtype=torch.float16
            )
    return tensors


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.to(torch.float16).cpu().numpy().tobytes()
    ).hexdigest()


def _group(targets, unit_for_book: str) -> dict:
    family, k = parse_format_name(_FMT)
    return {
        "format": _FMT,
        "targets": list(targets),
        "scheme": {
            "codebook_ref": _role_refs(unit_for_book),
            "codebook_source": "learned",
            "k": k,
            "mode": family.mode,
            "n_sub": family.n_sub,
        },
    }


def _tensors(*, passthrough: bool = False) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {
        f"{_PACKED}.cb_qweight": torch.zeros((4, 8, 16), dtype=torch.uint8),
        f"{_PACKED}.weight_scale": torch.ones((4, 8, 1), dtype=torch.float32),
        f"{_DENSE}.cb_qweight": torch.zeros((8, 16), dtype=torch.uint8),
        f"{_DENSE}.weight_scale": torch.ones((8, 1), dtype=torch.float32),
        # A plain float tensor, so `tensor_count` is not just the quantized ones.
        "model.norm.weight": torch.ones((8,), dtype=torch.bfloat16),
    }
    if passthrough:
        # Verbatim source bytes: an FP8 `.weight` with its block scale, produced
        # by no exporter here and declared rather than grouped.
        tensors[f"{_PASSTHROUGH}.weight"] = torch.zeros(
            (16, 8), dtype=torch.float8_e4m3fn
        )
        # E8M0, because that is the dtype the gate recognises as a scale plane
        # -- a passthrough unit whose scale it cannot see is a promise the
        # artifact cannot keep, and it says so.
        tensors[f"{_PASSTHROUGH}.weight_scale"] = torch.ones(
            (1, 1), dtype=torch.float32
        ).to(torch.float8_e8m0fnu)
    return tensors


def _quant_config(
    codebooks: dict[str, torch.Tensor],
    *,
    routed_targets: tuple[tuple[tuple[str, ...], str], ...],
    passthrough: bool = False,
    excluded: list[str] | None = None,
    overlay: bool = False,
) -> dict:
    groups: dict[str, dict] = {
        f"group_{index}": _group(targets, book)
        for index, (targets, book) in enumerate(routed_targets)
    }
    groups["group_dense"] = _group((_DENSE,), _DENSE)
    provenance: dict = {
        "renderer_abi": "prismaquant.nvfp4_cb_renderer.v1",
        "codebook_sha256": {
            ref: _digest(tensor) for ref, tensor in codebooks.items()
        },
    }
    if excluded is not None:
        provenance["excluded_namespaces"] = list(excluded)
    if overlay:
        provenance["dspark_source_overlay"] = {"schema": "test"}
    config: dict = {
        "format": "nvfp4_cb",
        "quant_method": "gridbook",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": groups,
        "ignore": ["model.norm"],
        "provenance": provenance,
    }
    if passthrough:
        config["source_passthrough"] = {
            "version": 1,
            "units": {_PASSTHROUGH: "fp8_e4m3_ue8m0_block128"},
        }
    return config


def _build(
    root: Path,
    *,
    routed_targets=(((_GATE,), _GATE), ((_UP,), _UP)),
    books=(_GATE, _UP, _DENSE),
    model_type: str = "qwen3",
    **kwargs,
) -> dict:
    codebooks = _codebooks(books)
    save_file(_tensors(passthrough=kwargs.get("passthrough", False)),
              str(root / "model.safetensors"))
    save_file(codebooks, str(root / "cb_codebooks.pqcb"))
    quant_config = _quant_config(
        codebooks, routed_targets=routed_targets, **kwargs
    )
    (root / "config.json").write_text(
        json.dumps({"model_type": model_type,
                    "architectures": ["Qwen3ForCausalLM"]}),
        encoding="utf-8",
    )
    (root / "quant_config.json").write_text(
        json.dumps(quant_config), encoding="utf-8"
    )
    return quant_config


# ---------------------------------------------------------------------------
# The per-role cover
# ---------------------------------------------------------------------------


def test_two_half_groups_cover_one_packed_stack(tmp_path, routed_profile):
    """The failure that refused a correct 87 GB body at its last gate."""

    quant_config = _build(tmp_path)
    evidence = cbv.validate_cb_artifact_decode_contract(tmp_path, quant_config)

    assert evidence["complete"] is True
    # Two roles, two books, ONE tensor: the group counts sum past the unit
    # count, which is exactly why the receipt's sum check became a bound.
    covered = {entry["group"]: entry["unit_count"]
               for entry in evidence["group_cover"]}
    assert covered == {"group_0": 1, "group_1": 1, "group_dense": 1}
    assert evidence["cb_unit_count"] == 2
    assert evidence["required_runtime_features"][
        cbv.CB_ROUTED_MOE_RUNTIME_FEATURE
    ] == cbv.CB_ROUTED_MOE_RUNTIME_FEATURE_VERSION


def test_one_half_leaves_the_other_role_with_no_book(tmp_path, routed_profile):
    """EVERY role, never any.

    Gridbook refuses to decode a stack that names no book for some role rather
    than reusing another role's; the checker has to refuse the same artifact,
    or it certifies something the runtime will reject.
    """

    quant_config = _build(tmp_path, routed_targets=(((_GATE,), _GATE),),
                          books=(_GATE, _DENSE))
    with pytest.raises(cbv.CBEndpointValidationError) as excinfo:
        cbv.validate_cb_artifact_decode_contract(tmp_path, quant_config)
    assert _PACKED in str(excinfo.value)


def test_the_packed_spelling_still_covers_every_role(tmp_path, routed_profile):
    """A lattice layer shares one book and legally names the packed stack."""

    quant_config = _build(
        tmp_path,
        routed_targets=(((_PACKED,), _PACKED),),
        books=(_PACKED, _DENSE),
    )
    evidence = cbv.validate_cb_artifact_decode_contract(tmp_path, quant_config)

    assert evidence["complete"] is True
    assert evidence["cb_unit_count"] == 2


def test_two_groups_claiming_one_role_are_ambiguous(tmp_path, routed_profile):
    """Per-role does not mean per-group-free-for-all.

    Refused even though the two groups here agree on their book, which
    Gridbook's `setdefault` would tolerate: stricter than the consumer is the
    fail-closed direction, and one role declared twice is an exporter bug.
    """

    quant_config = _build(
        tmp_path,
        routed_targets=(((_GATE,), _GATE), ((_UP,), _UP), ((_GATE,), _GATE)),
    )
    with pytest.raises(cbv.CBEndpointValidationError, match="claimed more than once"):
        cbv.validate_cb_artifact_decode_contract(tmp_path, quant_config)


def test_a_neighbouring_container_never_borrows_a_role(tmp_path, routed_profile):
    """`experts2` is not `experts`; the routed test is on dotted boundaries."""

    quant_config = _build(tmp_path)
    mutated = copy.deepcopy(quant_config)
    mutated["config_groups"]["group_1"]["targets"] = [
        _UP.replace(".experts.", ".experts2.")
    ]
    (tmp_path / "quant_config.json").write_text(
        json.dumps(mutated), encoding="utf-8"
    )
    with pytest.raises(cbv.CBEndpointValidationError):
        cbv.validate_cb_artifact_decode_contract(tmp_path, mutated)


# ---------------------------------------------------------------------------
# source_passthrough as a declared mechanism
# ---------------------------------------------------------------------------


def test_a_declared_passthrough_unit_is_not_unclaimed(tmp_path, routed_profile):
    """64 of the body's 336 passthrough units are named by no config group.

    The served 112.69 GB artifact has the same population, so this is a shipped
    shape rather than a defect -- the cover simply had never seen an FP8-source
    model.
    """

    quant_config = _build(tmp_path, passthrough=True)
    evidence = cbv.validate_cb_artifact_decode_contract(tmp_path, quant_config)

    assert evidence["complete"] is True
    assert evidence["passthrough_unit_count"] == 1
    # It is not counted as CB, and it is not counted as an embedding either.
    assert evidence["cb_unit_count"] == 2
    assert evidence["embedding_unit_count"] == 0


def test_an_undeclared_quantized_tensor_is_still_unclaimed(tmp_path, routed_profile):
    """The exemption is the declaration, not the shape of the tensor."""

    quant_config = _build(tmp_path, passthrough=True)
    mutated = copy.deepcopy(quant_config)
    mutated["source_passthrough"]["units"] = {}
    (tmp_path / "quant_config.json").write_text(
        json.dumps(mutated), encoding="utf-8"
    )
    with pytest.raises(cbv.CBEndpointValidationError) as excinfo:
        cbv.validate_cb_artifact_decode_contract(tmp_path, mutated)
    assert _PASSTHROUGH in str(excinfo.value)


# ---------------------------------------------------------------------------
# The three DSv4 topologies
# ---------------------------------------------------------------------------


def test_a_split_body_declares_itself_by_recording_the_omission(
    tmp_path, routed_profile
):
    quant_config = _build(
        tmp_path, model_type="deepseek_v4",
        excluded=[cbv.DSPARK_MTP_SOURCE_PREFIX],
    )
    evidence = cbv.validate_cb_artifact_decode_contract(tmp_path, quant_config)

    assert evidence["mode"] == cbv.CB_PLAIN_MODE
    assert evidence["excluded_namespaces"] == [cbv.DSPARK_MTP_SOURCE_PREFIX]


def test_a_dsv4_body_that_records_nothing_still_refuses(tmp_path, routed_profile):
    """The fault the guard was written for, unchanged.

    An artifact that LOST its overlay declares no topology and records no
    exclusion, which is indistinguishable from a body half only if the
    declaration is optional. It is not.
    """

    quant_config = _build(tmp_path, model_type="deepseek_v4")
    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="one of dspark_source_overlay or dspark_cb_sidecar",
    ):
        cbv.validate_cb_artifact_decode_contract(tmp_path, quant_config)


def test_an_overlay_cannot_also_exclude_the_namespace_it_is_built_from(
    tmp_path, routed_profile
):
    quant_config = _build(
        tmp_path, model_type="deepseek_v4", overlay=True,
        excluded=[cbv.DSPARK_MTP_SOURCE_PREFIX],
    )
    with pytest.raises(
        cbv.CBEndpointValidationError, match="constructed from exactly those"
    ):
        cbv.validate_cb_artifact_decode_contract(tmp_path, quant_config)


@pytest.mark.parametrize("model_type", ["qwen3", "deepseek_v4"])
def test_only_the_mtp_namespace_may_be_excluded(tmp_path, routed_profile, model_type):
    """A named permission, not an open field.

    The receipt replay has no disk and no lane, so the one omission a plain
    artifact may make has to be spelled there for the number to mean anything.
    """

    quant_config = _build(tmp_path, model_type=model_type, excluded=["visual."])
    with pytest.raises(cbv.CBEndpointValidationError):
        cbv.validate_cb_artifact_decode_contract(tmp_path, quant_config)
