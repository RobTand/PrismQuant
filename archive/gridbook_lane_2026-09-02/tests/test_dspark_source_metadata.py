"""Focused producer contract for DeepSeek-V4 DSpark source metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant.dspark_source_metadata import (
    FP8_BLOCK_UE8M0_SOURCE_FORMAT,
    MXFP4_SOURCE_FORMAT,
    apply_dspark_sidecar_overlay,
    build_dspark_sidecar_overlay,
    discover_dspark_source_overlay,
    discover_dspark_source_overlay_from_artifact,
)
from prismaquant.export_nvfp4_cb_streaming import export_nvfp4_cb_streaming


_EXPERTS = 2
_BODY_LAYERS = 3
_HIDDEN = 32
_HEADS = 2
_HEAD_DIM = 16
_Q_LORA_RANK = 16
_O_GROUPS = 2
_O_LORA_RANK = 8
_MOE_INTERMEDIATE = 32
_VOCAB = 64
_MARKOV_RANK = 8
_DENSE_REST_SHAPES = {
    "attn.wq_a": (_Q_LORA_RANK, _HIDDEN),
    "attn.wkv": (_HEAD_DIM, _HIDDEN),
    "attn.wq_b": (_HEADS * _HEAD_DIM, _Q_LORA_RANK),
    "attn.wo_a": (
        _O_GROUPS * _O_LORA_RANK,
        _HEADS * _HEAD_DIM // _O_GROUPS,
    ),
    "attn.wo_b": (_HIDDEN, _O_GROUPS * _O_LORA_RANK),
    "ffn.shared_experts.w1": (_MOE_INTERMEDIATE, _HIDDEN),
    "ffn.shared_experts.w2": (_HIDDEN, _MOE_INTERMEDIATE),
    "ffn.shared_experts.w3": (_MOE_INTERMEDIATE, _HIDDEN),
}
_PLAIN_MTP_BASES = {
    "mtp.0.ffn.gate",
    "mtp.1.ffn.gate",
    "mtp.2.confidence_head.proj",
    "mtp.2.ffn.gate",
    "mtp.2.markov_head.markov_w1",
    "mtp.2.markov_head.markov_w2",
}


class _MetadataSkeleton:
    def __init__(self, entries):
        self.entries = dict(entries)

    def keys(self):
        return self.entries.keys()

    def get_shape(self, name):
        return self.entries[name][0]

    def get_dtype(self, name):
        return self.entries[name][1]


def _source_config() -> dict:
    return {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": _HIDDEN,
        "num_hidden_layers": _BODY_LAYERS,
        "num_attention_heads": _HEADS,
        "head_dim": _HEAD_DIM,
        "q_lora_rank": _Q_LORA_RANK,
        "o_groups": _O_GROUPS,
        "o_lora_rank": _O_LORA_RANK,
        "moe_intermediate_size": _MOE_INTERMEDIATE,
        "n_routed_experts": _EXPERTS,
        "n_shared_experts": 1,
        "vocab_size": _VOCAB,
        "expert_dtype": "fp4",
        "dspark_block_size": 5,
        "dspark_markov_rank": _MARKOV_RANK,
        "dspark_target_layer_ids": [0, 1, 2],
        "num_nextn_predict_layers": 1,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "scale_fmt": "ue8m0",
        },
    }


def _metadata_entries() -> dict[str, tuple[tuple[int, ...], str]]:
    entries = {}
    for stage in range(3):
        for rest, shape in _DENSE_REST_SHAPES.items():
            base = f"mtp.{stage}.{rest}"
            entries[base + ".weight"] = (shape, "F8_E4M3")
            entries[base + ".scale"] = (
                tuple(-(-dimension // 128) for dimension in shape),
                "F8_E8M0",
            )
        for expert in range(_EXPERTS):
            for leaf, logical_shape in {
                "w1": (_MOE_INTERMEDIATE, _HIDDEN),
                "w2": (_HIDDEN, _MOE_INTERMEDIATE),
                "w3": (_MOE_INTERMEDIATE, _HIDDEN),
            }.items():
                base = f"mtp.{stage}.ffn.experts.{expert}.{leaf}"
                out_features, in_features = logical_shape
                entries[base + ".weight"] = (
                    (out_features, in_features // 2),
                    "I8",
                )
                entries[base + ".scale"] = (
                    (out_features, in_features // 32),
                    "F8_E8M0",
                )

        entries[f"mtp.{stage}.attn.q_norm.weight"] = (
            (_Q_LORA_RANK,), "BF16"
        )
        entries[f"mtp.{stage}.attn.kv_norm.weight"] = (
            (_HEAD_DIM,), "BF16"
        )
        for leaf in ("attn_norm.weight", "ffn_norm.weight"):
            entries[f"mtp.{stage}.{leaf}"] = ((_HIDDEN,), "BF16")
        entries[f"mtp.{stage}.ffn.gate.weight"] = (
            (_EXPERTS, _HIDDEN), "BF16"
        )
        entries[f"mtp.{stage}.ffn.gate.bias"] = ((_EXPERTS,), "F32")
        entries[f"mtp.{stage}.attn.attn_sink"] = ((_HEADS,), "F32")
        mix_hc = 24
        hc_dim = 4 * _HIDDEN
        for branch in ("attn", "ffn"):
            entries[f"mtp.{stage}.hc_{branch}_fn"] = (
                (mix_hc, hc_dim), "F32"
            )
            entries[f"mtp.{stage}.hc_{branch}_base"] = (
                (mix_hc,), "F32"
            )
            entries[f"mtp.{stage}.hc_{branch}_scale"] = ((3,), "F32")

    main_proj_shape = (_HIDDEN, _HIDDEN * 3)
    entries["mtp.0.main_proj.weight"] = (main_proj_shape, "F8_E4M3")
    entries["mtp.0.main_proj.scale"] = (
        tuple(-(-dimension // 128) for dimension in main_proj_shape),
        "F8_E8M0",
    )
    entries["mtp.0.main_norm.weight"] = ((_HIDDEN,), "BF16")
    entries["mtp.2.norm.weight"] = ((_HIDDEN,), "BF16")
    entries["mtp.2.confidence_head.proj.weight"] = (
        (1, _HIDDEN + _MARKOV_RANK), "BF16"
    )
    for leaf in ("markov_w1", "markov_w2"):
        entries[f"mtp.2.markov_head.{leaf}.weight"] = (
            (_VOCAB, _MARKOV_RANK), "BF16"
        )
    entries["mtp.2.hc_head_fn"] = ((4, 4 * _HIDDEN), "F32")
    entries["mtp.2.hc_head_base"] = ((4,), "F32")
    entries["mtp.2.hc_head_scale"] = ((1,), "F32")
    return entries


def _source_tensors() -> dict[str, torch.Tensor]:
    tensors = {}
    for name, (shape, dtype) in _metadata_entries().items():
        if dtype == "F8_E4M3":
            tensors[name] = torch.ones(shape).to(torch.float8_e4m3fn)
        elif dtype == "F8_E8M0":
            tensors[name] = torch.ones(shape).to(torch.float8_e8m0fnu)
        elif dtype == "I8":
            tensors[name] = torch.zeros(shape, dtype=torch.int8)
        elif dtype == "BF16":
            tensors[name] = torch.ones(shape, dtype=torch.bfloat16)
        elif dtype == "F32":
            tensors[name] = torch.ones(shape, dtype=torch.float32)
        else:  # pragma: no cover - fixture table is closed above
            raise AssertionError(dtype)
    tensors["model.norm.weight"] = torch.ones(
        _HIDDEN, dtype=torch.bfloat16
    )
    return tensors


def _write_source(
    root: Path, *, config: dict | None = None
) -> dict[str, torch.Tensor]:
    root.mkdir()
    tensors = _source_tensors()
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text(
        json.dumps(_source_config() if config is None else config)
    )
    return tensors


def _export(
    root: Path,
    *,
    config: dict | None = None,
    **export_kwargs,
) -> tuple[Path, Path, dict[str, torch.Tensor]]:
    source = root / "source"
    tensors = _write_source(source, config=config)
    assignment = root / "assignment.json"
    assignment.write_text("{}")
    artifact = root / "artifact"
    export_nvfp4_cb_streaming(
        source,
        assignment,
        artifact,
        {},
        device="cpu",
        allow_unstamped_research=True,
        allow_route_pending_passthrough=True,
        **export_kwargs,
    )
    return source, artifact, tensors


def _source_target_groups(quant_config: dict) -> dict[str, set[str]]:
    out = {}
    for group in quant_config["config_groups"].values():
        format_name = group.get("source_format")
        if format_name in {
            FP8_BLOCK_UE8M0_SOURCE_FORMAT,
            MXFP4_SOURCE_FORMAT,
        }:
            out[format_name] = set(group["targets"])
    return out


def test_discovery_maps_physical_targets_to_construction_namespace():
    overlay = discover_dspark_source_overlay(
        _MetadataSkeleton(_metadata_entries()), _source_config()
    )
    assert overlay is not None
    assert overlay.n_mtp_layers == 3
    assert len(overlay.physical_targets) == 43
    assert len(overlay.construction_units) == 22
    assert overlay.physical_targets["mtp.0.main_proj"] == (
        FP8_BLOCK_UE8M0_SOURCE_FORMAT
    )
    assert overlay.physical_to_construction_unit[
        "mtp.1.ffn.experts.1.w3"
    ] == "model.layers.4.ffn.experts"
    assert overlay.physical_to_construction_unit["mtp.2.attn.wkv"] == (
        "model.layers.5.attn.fused_wqa_wkv"
    )


def test_discovery_refuses_incomplete_three_stage_contract():
    entries = _metadata_entries()
    entries.pop("mtp.1.ffn.experts.1.w3.scale")
    with pytest.raises(ValueError, match="requires both .weight and .scale"):
        discover_dspark_source_overlay(
            _MetadataSkeleton(entries), _source_config()
        )


@pytest.mark.parametrize(
    "missing_name",
    (
        "mtp.1.ffn.gate.weight",
        "mtp.2.confidence_head.proj.weight",
        "mtp.0.attn.q_norm.weight",
        "mtp.2.hc_head_scale",
    ),
)
def test_discovery_refuses_missing_essential_glue(missing_name):
    entries = _metadata_entries()
    entries.pop(missing_name)
    with pytest.raises(ValueError, match="essential tensor layout.*missing"):
        discover_dspark_source_overlay(
            _MetadataSkeleton(entries), _source_config()
        )


def test_discovery_refuses_unknown_essential_glue():
    entries = _metadata_entries()
    entries["mtp.1.hc_attn_future"] = ((24,), "F32")
    with pytest.raises(ValueError, match="essential tensor layout.*unknown"):
        discover_dspark_source_overlay(
            _MetadataSkeleton(entries), _source_config()
        )


@pytest.mark.parametrize(
    ("name", "replacement"),
    (
        ("mtp.0.attn.wq_a.weight", ((15, _HIDDEN), "F8_E4M3")),
        ("mtp.1.ffn.experts.0.w2.scale", ((_HIDDEN, 1), "BF16")),
        ("mtp.2.ffn.gate.weight", ((_EXPERTS, _HIDDEN + 1), "BF16")),
        ("mtp.0.attn.kv_norm.weight", ((_HEAD_DIM,), "F32")),
        ("mtp.2.hc_head_fn", ((4, 4 * _HIDDEN + 1), "F32")),
    ),
)
def test_discovery_refuses_bad_released_shapes_and_dtypes(name, replacement):
    entries = _metadata_entries()
    entries[name] = replacement
    with pytest.raises(ValueError, match="released DSpark tensor must be"):
        discover_dspark_source_overlay(
            _MetadataSkeleton(entries), _source_config()
        )


@pytest.mark.parametrize(
    "target_ids",
    (
        [0, 0, 2],
        [-1, 1, 2],
        [0, 1, _BODY_LAYERS],
    ),
)
def test_discovery_refuses_duplicate_or_out_of_range_target_ids(target_ids):
    config = _source_config()
    config["dspark_target_layer_ids"] = target_ids
    with pytest.raises(ValueError, match="distinct and each in"):
        discover_dspark_source_overlay(
            _MetadataSkeleton(_metadata_entries()), config
        )


def test_released_fixture_covers_every_nonquantized_dspark_tensor_class():
    entries = _metadata_entries()
    assert len(entries) == 2 * 43 + 6 + 14 + 27
    assert sum(dtype == "BF16" for _shape, dtype in entries.values()) == 20
    assert sum(dtype == "F32" for _shape, dtype in entries.values()) == 27


def test_streaming_export_emits_metadata_overlay_without_reencoding_mtp(
    tmp_path,
):
    _source, artifact, source_tensors = _export(tmp_path)
    config = json.loads((artifact / "config.json").read_text())
    quant = json.loads((artifact / "quant_config.json").read_text())
    overlay = discover_dspark_source_overlay_from_artifact(artifact)
    assert overlay is not None
    assert config["n_mtp_layers"] == 3
    assert quant["provenance"]["dspark_source_overlay"] == overlay.provenance()
    assert set(quant["ignore"]) & set(overlay.physical_targets) == set()
    assert {value for value in quant["ignore"] if value.startswith("mtp.")} == (
        _PLAIN_MTP_BASES
    )
    assert len(_source_target_groups(quant)[MXFP4_SOURCE_FORMAT]) == 18
    assert len(_source_target_groups(quant)[FP8_BLOCK_UE8M0_SOURCE_FORMAT]) == 25
    units = quant["source_passthrough"]["units"]
    assert len(units) == 22
    assert units["model.layers.3.ffn.experts"] == (
        "mxfp4_e2m1_ue8m0_g32"
    )

    emitted = load_file(str(artifact / "model.safetensors"))
    assert set(emitted) == set(source_tensors)
    for name, source_tensor in source_tensors.items():
        assert name in emitted
        if source_tensor.dtype in {
            torch.float8_e4m3fn,
            torch.float8_e8m0fnu,
        }:
            assert torch.equal(
                emitted[name].view(torch.uint8),
                source_tensor.view(torch.uint8),
            )
        else:
            assert torch.equal(emitted[name], source_tensor)


def test_dspark_cb_sidecar_flag_off_preserves_target_golden_inventory(
    tmp_path,
):
    """Flag-off retains the source checkpoint's physical target contract."""

    default_root = tmp_path / "default"
    explicit_root = tmp_path / "explicit-false"
    default_root.mkdir()
    explicit_root.mkdir()
    source_root_a, artifact_a, _source_a = _export(default_root)
    source_root_b, artifact_b, _source_b = _export(
        explicit_root, dspark_cb_sidecar=False
    )
    # This is the compatibility authority: two untouched source checkpoints,
    # created before either target artifact is read.  The assertion is not an
    # omitted-vs-false self-consistency check; each export must preserve the
    # complete physical inventory and exact payload bytes of its own source.
    golden_a = load_file(str(source_root_a / "model.safetensors"))
    golden_b = load_file(str(source_root_b / "model.safetensors"))
    emitted_a = load_file(str(artifact_a / "model.safetensors"))
    emitted_b = load_file(str(artifact_b / "model.safetensors"))
    golden_inventory = {
        name: (tuple(tensor.shape), tensor.dtype)
        for name, tensor in golden_a.items()
    }

    assert {
        name: (tuple(tensor.shape), tensor.dtype)
        for name, tensor in golden_b.items()
    } == golden_inventory
    assert {
        name: (tuple(tensor.shape), tensor.dtype)
        for name, tensor in emitted_a.items()
    } == golden_inventory
    assert {
        name: (tuple(tensor.shape), tensor.dtype)
        for name, tensor in emitted_b.items()
    } == golden_inventory
    for name in sorted(golden_inventory):
        assert torch.equal(
            emitted_a[name].view(torch.uint8),
            golden_a[name].view(torch.uint8),
        )
        assert torch.equal(
            emitted_b[name].view(torch.uint8),
            golden_b[name].view(torch.uint8),
        )
        assert torch.equal(
            emitted_a[name].view(torch.uint8),
            emitted_b[name].view(torch.uint8),
        )
    assert (artifact_a / "config.json").read_bytes() == (
        artifact_b / "config.json"
    ).read_bytes()
    expected_config = _source_config()
    expected_config["quantization_config"] = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "config_file": "quant_config.json",
    }
    expected_config["n_mtp_layers"] = 3
    assert json.loads((artifact_a / "config.json").read_text()) == (
        expected_config
    )
    assert (artifact_a / "quant_config.json").read_bytes() == (
        artifact_b / "quant_config.json"
    ).read_bytes()
    quant = json.loads((artifact_a / "quant_config.json").read_text())
    assert "dspark_cb_sidecar" not in quant["provenance"]
    assert "dspark_render_attestation" not in quant["provenance"]
    assert quant["provenance"]["dspark_source_overlay"]["tensor_bytes_rewritten"] == 0


@pytest.mark.parametrize(
    "export_kwargs",
    (
        {"subset_prefixes": ["model."]},
        {"exclude_namespaces": ["mtp."]},
    ),
)
def test_body_only_export_strips_stale_dspark_layer_stamp(
    tmp_path, export_kwargs
):
    config = _source_config()
    config["n_mtp_layers"] = 3
    _source, artifact, _source_tensors_dict = _export(
        tmp_path, config=config, **export_kwargs
    )

    emitted_config = json.loads((artifact / "config.json").read_text())
    emitted_quant = json.loads((artifact / "quant_config.json").read_text())
    emitted = load_file(str(artifact / "model.safetensors"))
    assert "n_mtp_layers" not in emitted_config
    assert "dspark_source_overlay" not in emitted_quant["provenance"]
    assert not any(name.startswith("mtp.") for name in emitted)


@pytest.mark.parametrize(
    "export_kwargs",
    (
        {"subset_prefixes": ["mtp.0."]},
        {"exclude_namespaces": ["mtp.0."]},
    ),
)
def test_partial_dspark_subset_or_exclusion_is_refused(
    tmp_path, export_kwargs
):
    source = tmp_path / "source"
    _write_source(source)
    assignment = tmp_path / "assignment.json"
    assignment.write_text("{}")
    with pytest.raises(ValueError, match="partial DSpark"):
        export_nvfp4_cb_streaming(
            source,
            assignment,
            tmp_path / "artifact",
            {},
            device="cpu",
            allow_unstamped_research=True,
            allow_route_pending_passthrough=True,
            **export_kwargs,
        )


def test_sidecar_overlay_preserves_model_sha_inode_mtime_and_body_metadata(
    tmp_path,
):
    _source, artifact, _source_tensors_dict = _export(tmp_path)
    overlay = discover_dspark_source_overlay_from_artifact(artifact)
    assert overlay is not None

    # Recreate the pre-overlay artifact state whose tensor bytes are already
    # correct: physical MTP targets ignored, no construction routes/stamp, and
    # no n_mtp_layers.  Its inventory is intentionally stale; applying the
    # sidecar repairs that self-sized producer record as part of the transaction.
    config_path = artifact / "config.json"
    quant_path = artifact / "quant_config.json"
    config = json.loads(config_path.read_text())
    quant = json.loads(quant_path.read_text())
    config.pop("n_mtp_layers")
    for group in quant["config_groups"].values():
        group["targets"] = [
            target for target in group.get("targets", [])
            if "mtp[.]" not in target
        ]
    quant["config_groups"] = {
        key: group for key, group in quant["config_groups"].items()
        if group.get("targets")
    }
    quant["ignore"] = sorted(set(quant["ignore"]) | set(overlay.physical_targets))
    quant["source_passthrough"]["units"] = {
        unit: wire
        for unit, wire in quant["source_passthrough"]["units"].items()
        if unit not in overlay.construction_units
    }
    if not quant["source_passthrough"]["units"]:
        quant.pop("source_passthrough")
    quant["provenance"].pop("dspark_source_overlay")
    quant["provenance"]["source_passthrough_targets"] = {}
    config_path.write_text(json.dumps(config, indent=2))
    quant_path.write_text(json.dumps(quant, indent=2))

    # The pure planner is read-only and body-neutral.
    before_sidecars = (config_path.read_bytes(), quant_path.read_bytes())
    planned_config, planned_quant = build_dspark_sidecar_overlay(
        config, quant, overlay
    )
    assert (config_path.read_bytes(), quant_path.read_bytes()) == before_sidecars
    assert planned_config["n_mtp_layers"] == 3
    assert planned_quant["quant_method"] == quant["quant_method"]

    model_path = artifact / "model.safetensors"
    before_stat = model_path.stat()
    before_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    published = tmp_path / "artifact-dspark"
    applied = apply_dspark_sidecar_overlay(artifact, published)
    after_stat = model_path.stat()
    after_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert applied == overlay
    assert after_sha == before_sha
    assert after_stat.st_ino == before_stat.st_ino
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert (config_path.read_bytes(), quant_path.read_bytes()) == before_sidecars
    published_model = published / "model.safetensors"
    assert published_model.stat().st_ino == before_stat.st_ino
    assert hashlib.sha256(published_model.read_bytes()).hexdigest() == before_sha

    final_config_path = published / "config.json"
    final_quant_path = published / "quant_config.json"
    final_config = json.loads(final_config_path.read_text())
    final_quant = json.loads(final_quant_path.read_text())
    assert final_config["n_mtp_layers"] == 3
    assert len(final_quant["ignore"]) + len(overlay.physical_targets) == len(
        quant["ignore"]
    )
    inventory = final_quant["provenance"]["artifact_inventory"]
    assert inventory["file_bytes"]["config.json"] == final_config_path.stat().st_size
    assert inventory["file_bytes"]["quant_config.json"] == (
        final_quant_path.stat().st_size
    )
