"""CPU coverage for resident routed-MoE learned per-role book wiring."""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant import format_registry as fr
from prismaquant import nvfp4_cb_formats as cb
from prismaquant import cb_learned_bundle
from prismaquant import gridbook_runtime_pin as runtime_pin
from prismaquant.cb_learned_bundle import train_and_save_bundle
from prismaquant.nvfp4_cb_footprint import (
    CB_TENSOR_IDENTITY_FIELD,
    cb_assignment_serialization_stamps,
    cb_serialization_context_from_env,
)
from prismaquant.production_weight_cache import (
    bind_cb_render_identity_source_weights,
    build_production_cache_cb_render_identity,
)


EXPERTS = 2
ROWS = 4
WIDTH = 256
PACKED = "model.layers.0.mlp.experts.gate_up_proj"
PACKED_DOWN = "model.layers.0.mlp.experts.down_proj"
GATE = "model.layers.0.mlp.experts.gate_proj"
UP = "model.layers.0.mlp.experts.up_proj"
DOWN = "model.layers.0.mlp.experts.down_proj"
FORMAT = "FP8_CBL_K28"
FORMAT_32 = "FP8_CBL_K32"


class _PerExpertProfile:
    def per_expert_moe_regex(self):
        return (
            r"re:^model[.]layers[.][0-9]+[.]mlp[.]experts[.]"
            r"[0-9]+[.](gate|up|down)_proj$"
        )

    def packed_expert_param_names(self):
        return frozenset({"gate_up_proj", "down_proj"})

    def packed_expert_parent_for_projection(self, projection):
        return {
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
            "down_proj": "down_proj",
        }.get(projection)

    def packed_expert_projection_names(self, parent):
        return {
            "gate_up_proj": ("gate_proj", "up_proj"),
            "down_proj": ("down_proj",),
        }[parent]

    def vllm_fused_moe_scheme_projection_names(self, parent):
        return self.packed_expert_projection_names(parent)

    def checkpoint_to_live_name(self, name, *, multimodal=False):
        del multimodal
        return name

    def source_tensor_name(self, name):
        return name

    def fused_sibling_group(self, name):
        del name
        return None

    def to_vllm_internal_name(self, name):
        return name


@pytest.fixture(autouse=True)
def _pretend_gridbook_supports_routed_lut(monkeypatch):
    """Patch the pin to a version that carries the routed per-role LUT ABI.

    These tests exercise the routed learned-book PRODUCER, which is gated on
    ``GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION``. The shipped pin
    is the released 0.8.2, which correctly refuses that path — no released
    Gridbook carries the ABI yet. Reading the shipped pin here would make the
    producer's own tests silently vanish the day the pin is right and fail the
    day it is not, so supply the capability explicitly instead.
    """
    supported = runtime_pin.parse_gridbook_runtime_pin({
        "schema": runtime_pin.GRIDBOOK_RUNTIME_PIN_SCHEMA,
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "a" * 40,
        "version": (
            runtime_pin.GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION),
        "version_is_release": False,
    })
    monkeypatch.setattr(
        cb_learned_bundle, "load_gridbook_runtime_pin", lambda: supported)


def _set_learned_env(monkeypatch, bundle_path):
    monkeypatch.setenv("CB_CODEBOOK_SOURCE_SCOPE", "fp8")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "learned")
    monkeypatch.setenv("CB_CODEBOOK_BUNDLE", str(bundle_path))
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_SCALE_SWEEP", "1")
    monkeypatch.setenv("CB_SCALE_SWEEP_SCOPE", "all")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "0")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_SCOPE", "none")
    monkeypatch.setenv("PRISMAQUANT_CB_MINCHAIN", "0")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "balanced")


def _layer_config(
    path,
    physical_weight,
    physical_cw,
    down_weight,
    down_cw,
    context,
):
    assignment = {PACKED: FORMAT, PACKED_DOWN: FORMAT}
    source_weights = {PACKED: physical_weight, PACKED_DOWN: down_weight}
    physical_col_weights = {PACKED: physical_cw, PACKED_DOWN: down_cw}
    identity = build_production_cache_cb_render_identity(
        {qname: (FORMAT,) for qname in assignment},
        cb_serialization_context=context,
        col_weights=physical_col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    assert identity is not None
    identity = bind_cb_render_identity_source_weights(
        identity, source_weights
    )
    stamps = cb_assignment_serialization_stamps(
        assignment,
        {qname: tuple(weight.shape) for qname, weight in source_weights.items()},
        context=context,
    )
    payload = {
        qname: {
            **fr.get_format(FORMAT).autoround_config(),
            CB_TENSOR_IDENTITY_FIELD: stamps[qname],
        }
        for qname in assignment
    }
    payload["__prismaquant__"] = {
        "schema": "prismaquant.layer_config_meta.v1",
        "cb_serialized_payload": identity["cb_serialized_payload"],
        "cb_render_identity": identity,
    }
    path.write_text(json.dumps(payload))
    return path


def _fixture(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(1204)
    gate_members = []
    up_members = []
    down_members = []
    tensors = {"model.norm.weight": torch.ones(8, dtype=torch.bfloat16)}
    col_weights: dict[str, torch.Tensor] = {}
    gate_cws = []
    up_cws = []
    down_cws = []
    for expert_id in range(EXPERTS):
        gate_name = f"model.layers.0.mlp.experts.{expert_id}.gate_proj"
        up_name = f"model.layers.0.mlp.experts.{expert_id}.up_proj"
        down_name = f"model.layers.0.mlp.experts.{expert_id}.down_proj"
        gate = torch.randn(ROWS, WIDTH, generator=generator).to(torch.bfloat16)
        up = (torch.randn(ROWS, WIDTH, generator=generator) + 0.75).to(
            torch.bfloat16
        )
        down = (torch.randn(ROWS, WIDTH, generator=generator) - 0.5).to(
            torch.bfloat16
        )
        gate_members.append(gate)
        up_members.append(up)
        down_members.append(down)
        tensors[gate_name + ".weight"] = gate
        tensors[up_name + ".weight"] = up
        tensors[down_name + ".weight"] = down
        gate_cw = torch.linspace(0.1, 1.0, WIDTH) + expert_id
        up_cw = torch.linspace(1.1, 2.0, WIDTH) + expert_id
        down_cw = torch.linspace(2.1, 3.0, WIDTH) + expert_id
        col_weights[gate_name] = gate_cw
        col_weights[up_name] = up_cw
        col_weights[down_name] = down_cw
        gate_cws.append(gate_cw.reshape(1, -1))
        up_cws.append(up_cw.reshape(1, -1))
        down_cws.append(down_cw.reshape(1, -1))

    gate_stack = torch.stack(gate_members)
    up_stack = torch.stack(up_members)
    down_stack = torch.stack(down_members)
    physical = torch.cat((gate_stack, up_stack), dim=1)
    gate_cw_stack = torch.stack(gate_cws)
    up_cw_stack = torch.stack(up_cws)
    down_cw_stack = torch.stack(down_cws)
    physical_cw = (gate_cw_stack + up_cw_stack) * 0.5
    col_weights[PACKED] = physical_cw
    col_weights[PACKED_DOWN] = down_cw_stack

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file(tensors, str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["TestForCausalLM"]
    }))

    lattice = cb._resolve_codebook(
        28, "fp8", "product", None, torch.device("cpu")
    )
    lattice_32 = cb._resolve_codebook(
        32, "fp8", "product", None, torch.device("cpu")
    )
    gate_book = tuple(table.clone() for table in lattice)
    up_book = tuple(table.flip(0).contiguous() for table in lattice)
    down_book = tuple(table.flip(1).contiguous() for table in lattice)
    gate_book_32 = tuple(table.flip(1).contiguous() for table in lattice_32)
    up_book_32 = tuple(table.flip(0).contiguous() for table in lattice_32)
    down_book_32 = tuple(table.clone() for table in lattice_32)
    bundle_path = tmp_path / "learned.pqcb"

    def input_aliases(qname, weight, _cw):
        projection = qname.rsplit(".", 1)[-1]
        return {
            f"model.layers.0.mlp.experts.{expert_id}.{projection}": (
                weight[expert_id],
                col_weights[
                    f"model.layers.0.mlp.experts.{expert_id}.{projection}"
                ],
            )
            for expert_id in range(EXPERTS)
        }

    bundle = train_and_save_bundle(
        bundle_path,
        weights={GATE: gate_stack, UP: up_stack, DOWN: down_stack},
        col_weights={
            GATE: gate_cw_stack,
            UP: up_cw_stack,
            DOWN: down_cw_stack,
        },
        formats={
            GATE: (FORMAT, FORMAT_32),
            UP: (FORMAT, FORMAT_32),
            DOWN: (FORMAT, FORMAT_32),
        },
        learned_formats=(FORMAT, FORMAT_32),
        routed_moe_qnames=(GATE, UP, DOWN),
        pretrained_codebooks={
            (GATE, FORMAT): gate_book,
            (UP, FORMAT): up_book,
            (DOWN, FORMAT): down_book,
            (GATE, FORMAT_32): gate_book_32,
            (UP, FORMAT_32): up_book_32,
            (DOWN, FORMAT_32): down_book_32,
        },
        input_alias_provider=input_aliases,
    )
    _set_learned_env(monkeypatch, bundle_path)
    context = cb_serialization_context_from_env()
    config_path = _layer_config(
        tmp_path / "layer_config.json",
        physical,
        physical_cw,
        down_stack,
        down_cw_stack,
        context,
    )
    return {
        "model_dir": model_dir,
        "bundle": bundle,
        "config_path": config_path,
        "col_weights": col_weights,
        "gate_stack": gate_stack,
        "up_stack": up_stack,
        "down_stack": down_stack,
        "gate_cw": gate_cw_stack,
        "up_cw": up_cw_stack,
        "down_cw": down_cw_stack,
        "physical": physical,
        "physical_cw": physical_cw,
        "physical_down": down_stack,
        "physical_down_cw": down_cw_stack,
    }


def _streaming_layer_config(
    path, fixture, context, *, formats_by_member=None
):
    assignment = {
        name: (
            FORMAT
            if formats_by_member is None
            else formats_by_member[name]
        )
        for name in fixture["col_weights"]
        if name not in {PACKED, PACKED_DOWN}
    }
    source_weights = {}
    for projection, stack_key in (
        ("gate_proj", "gate_stack"),
        ("up_proj", "up_stack"),
        ("down_proj", "down_stack"),
    ):
        source_weights.update({
            f"model.layers.0.mlp.experts.{expert_id}.{projection}":
                fixture[stack_key][expert_id]
            for expert_id in range(EXPERTS)
        })
    member_col = {
        name: fixture["col_weights"][name]
        for name in assignment
    }
    identity = build_production_cache_cb_render_identity(
        {name: (format_name,) for name, format_name in assignment.items()},
        cb_serialization_context=context,
        col_weights=member_col,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    assert identity is not None
    identity = bind_cb_render_identity_source_weights(identity, source_weights)
    stamps = cb_assignment_serialization_stamps(
        assignment,
        {name: tuple(weight.shape) for name, weight in source_weights.items()},
        context=context,
    )
    payload = {
        name: {
            **fr.get_format(format_name).autoround_config(),
            CB_TENSOR_IDENTITY_FIELD: stamps[name],
        }
        for name, format_name in assignment.items()
    }
    payload["__prismaquant__"] = {
        "schema": "prismaquant.layer_config_meta.v1",
        "cb_serialized_payload": identity["cb_serialized_payload"],
        "cb_render_identity": identity,
    }
    path.write_text(json.dumps(payload))
    return path


def test_resident_export_encodes_fused_rows_with_distinct_role_books(
    tmp_path, monkeypatch
):
    from prismaquant import model_profiles
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        model_profiles, "detect_profile", lambda *_args, **_kwargs: _PerExpertProfile()
    )
    out = tmp_path / "out"
    export_nvfp4_cb(
        fixture["model_dir"],
        fixture["config_path"],
        out,
        fixture["col_weights"],
        shared_codebook_spec={"source": "learned"},
        device="cpu",
    )

    bundle = fixture["bundle"]
    gate_book = bundle.codebook_for(
        GATE,
        FORMAT,
        weight=fixture["gate_stack"],
        col_weights=fixture["gate_cw"],
    )
    up_book = bundle.codebook_for(
        UP,
        FORMAT,
        weight=fixture["up_stack"],
        col_weights=fixture["up_cw"],
    )
    down_book = bundle.codebook_for(
        DOWN,
        FORMAT,
        weight=fixture["down_stack"],
        col_weights=fixture["down_cw"],
    )
    gate_packed, gate_fields = cb.nvfp4_cb_pack(
        fixture["gate_stack"],
        28,
        grid="fp8",
        mode="product",
        col_weights=fixture["gate_cw"],
        codebook=gate_book,
        scale_sweep=True,
        scale_coding=cb.SCALE_CODING_V1,
        encode_tier="balanced",
    )
    up_packed, up_fields = cb.nvfp4_cb_pack(
        fixture["up_stack"],
        28,
        grid="fp8",
        mode="product",
        col_weights=fixture["up_cw"],
        codebook=up_book,
        scale_sweep=True,
        scale_coding=cb.SCALE_CODING_V1,
        encode_tier="balanced",
    )
    down_packed, down_fields = cb.nvfp4_cb_pack(
        fixture["down_stack"],
        28,
        grid="fp8",
        mode="product",
        col_weights=fixture["down_cw"],
        codebook=down_book,
        scale_sweep=True,
        scale_coding=cb.SCALE_CODING_V1,
        encode_tier="balanced",
    )
    expected = torch.cat((
        gate_packed.reshape(EXPERTS, ROWS, -1),
        up_packed.reshape(EXPERTS, ROWS, -1),
    ), dim=1).to(torch.uint8)
    emitted = load_file(str(out / "model.safetensors"))
    assert torch.equal(emitted[PACKED + ".cb_qweight"], expected)
    expected_scales = torch.cat((
        gate_fields["scales"].reshape(EXPERTS, ROWS),
        up_fields["scales"].reshape(EXPERTS, ROWS),
    ), dim=1).to(torch.float32)
    assert torch.equal(emitted[PACKED + ".weight_scale"], expected_scales)
    assert torch.equal(
        emitted[PACKED_DOWN + ".cb_qweight"],
        down_packed.reshape(EXPERTS, ROWS, -1).to(torch.uint8),
    )
    assert torch.equal(
        emitted[PACKED_DOWN + ".weight_scale"],
        down_fields["scales"].reshape(EXPERTS, ROWS).to(torch.float32),
    )

    quant_config = json.loads((out / "quant_config.json").read_text())
    groups = [
        group for group in quant_config["config_groups"].values()
        if group.get("format") == FORMAT
    ]
    assert {tuple(group["targets"]) for group in groups} == {
        (GATE,), (UP,), (DOWN,)
    }
    refs_by_target = {
        group["targets"][0]: tuple(group["scheme"]["codebook_ref"])
        for group in groups
    }
    assert refs_by_target[GATE] != refs_by_target[UP]
    assert len(set(refs_by_target.values())) == 3
    sidecar = load_file(str(out / "cb_codebooks.pqcb"))
    assert set(sidecar) == set().union(*map(set, refs_by_target.values()))


def test_resident_learned_direct_stack_without_member_map_fails_closed(
    tmp_path, monkeypatch
):
    from prismaquant import model_profiles
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    fixture = _fixture(tmp_path, monkeypatch)
    direct_dir = tmp_path / "direct"
    direct_dir.mkdir()
    save_file(
        {
            PACKED + ".weight": torch.cat((
                fixture["gate_stack"], fixture["up_stack"]
            ), dim=1),
            PACKED_DOWN + ".weight": fixture["down_stack"],
        },
        str(direct_dir / "model.safetensors"),
    )
    (direct_dir / "config.json").write_text("{}")
    monkeypatch.setattr(
        model_profiles, "detect_profile", lambda *_args, **_kwargs: _PerExpertProfile()
    )
    with pytest.raises(ValueError, match="already-packed rank-3 source"):
        export_nvfp4_cb(
            direct_dir,
            fixture["config_path"],
            tmp_path / "direct-out",
            fixture["col_weights"],
            shared_codebook_spec={"source": "learned"},
            device="cpu",
        )


def test_streaming_export_collapses_members_but_keeps_distinct_role_books(
    tmp_path, monkeypatch
):
    import importlib

    stream = importlib.import_module(
        "prismaquant.export_nvfp4_cb_streaming"
    )
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        stream, "detect_profile", lambda *_args, **_kwargs: _PerExpertProfile()
    )
    context = cb_serialization_context_from_env()
    config_path = _streaming_layer_config(
        tmp_path / "streaming-layer-config.json", fixture, context
    )
    out = tmp_path / "streaming-out"
    stream.export_nvfp4_cb_streaming(
        fixture["model_dir"],
        config_path,
        out,
        fixture["col_weights"],
        shared_codebook_spec={"source": "learned"},
        device="cpu",
    )

    bundle = fixture["bundle"]
    expected_parts = []
    expected_scales = []
    for qname, weight, col_weight in (
        (GATE, fixture["gate_stack"], fixture["gate_cw"]),
        (UP, fixture["up_stack"], fixture["up_cw"]),
    ):
        packed, fields = cb.nvfp4_cb_pack(
            weight,
            28,
            grid="fp8",
            mode="product",
            col_weights=col_weight,
            codebook=bundle.codebook_for(
                qname, FORMAT, weight=weight, col_weights=col_weight
            ),
            scale_sweep=True,
            scale_coding=cb.SCALE_CODING_V1,
            encode_tier="balanced",
        )
        expected_parts.append(packed.reshape(EXPERTS, ROWS, -1))
        expected_scales.append(fields["scales"].reshape(EXPERTS, ROWS))
    emitted = load_file(str(out / "model.safetensors"))
    assert torch.equal(
        emitted[PACKED + ".cb_qweight"],
        torch.cat(expected_parts, dim=1).to(torch.uint8),
    )
    assert torch.equal(
        emitted[PACKED + ".weight_scale"],
        torch.cat(expected_scales, dim=1).to(torch.float32),
    )
    down_packed, down_fields = cb.nvfp4_cb_pack(
        fixture["down_stack"],
        28,
        grid="fp8",
        mode="product",
        col_weights=fixture["down_cw"],
        codebook=bundle.codebook_for(
            DOWN,
            FORMAT,
            weight=fixture["down_stack"],
            col_weights=fixture["down_cw"],
        ),
        scale_sweep=True,
        scale_coding=cb.SCALE_CODING_V1,
        encode_tier="balanced",
    )
    assert torch.equal(
        emitted[PACKED_DOWN + ".cb_qweight"],
        down_packed.reshape(EXPERTS, ROWS, -1).to(torch.uint8),
    )
    assert torch.equal(
        emitted[PACKED_DOWN + ".weight_scale"],
        down_fields["scales"].reshape(EXPERTS, ROWS).to(torch.float32),
    )
    config = json.loads((out / "quant_config.json").read_text())
    learned_groups = [
        group for group in config["config_groups"].values()
        if group.get("format") == FORMAT
    ]
    assert {tuple(group["targets"]) for group in learned_groups} == {
        (GATE,), (UP,), (DOWN,)
    }
    assert len({
        tuple(group["scheme"]["codebook_ref"])
        for group in learned_groups
    }) == 3


def test_streaming_split_format_groups_reuse_exact_role_rung_books(
    tmp_path, monkeypatch
):
    """A mixed expert bank keeps both its rung and gate/up role identity."""

    import importlib

    stream = importlib.import_module(
        "prismaquant.export_nvfp4_cb_streaming"
    )
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        stream, "detect_profile", lambda *_args, **_kwargs: _PerExpertProfile()
    )
    context = cb_serialization_context_from_env()
    formats_by_member = {
        name: (FORMAT if ".experts.0." in name else FORMAT_32)
        for name in fixture["col_weights"]
        if name not in {PACKED, PACKED_DOWN}
    }
    config_path = _streaming_layer_config(
        tmp_path / "split-layer-config.json",
        fixture,
        context,
        formats_by_member=formats_by_member,
    )
    per_expert_path = tmp_path / "per-expert.json"
    per_expert_path.write_text(json.dumps({
        name: fr.get_format(format_name).autoround_config()
        for name, format_name in formats_by_member.items()
    }))
    out = tmp_path / "split-out"
    stream.export_nvfp4_cb_streaming(
        fixture["model_dir"],
        config_path,
        out,
        fixture["col_weights"],
        shared_codebook_spec={"source": "learned"},
        device="cpu",
        per_expert_config_path=per_expert_path,
    )

    emitted = load_file(str(out / "model.safetensors"))
    bundle = fixture["bundle"]
    for expert_id, format_name, rung in (
        (0, FORMAT, 28),
        (1, FORMAT_32, 32),
    ):
        expected_parts = []
        expected_scales = []
        for qname, stack_key, cw_key in (
            (GATE, "gate_stack", "gate_cw"),
            (UP, "up_stack", "up_cw"),
        ):
            role_weight = fixture[stack_key][expert_id:expert_id + 1]
            role_cw = fixture[cw_key][expert_id:expert_id + 1]
            packed, fields = cb.nvfp4_cb_pack(
                role_weight,
                rung,
                grid="fp8",
                mode="product",
                col_weights=role_cw,
                codebook=bundle.codebook_for(qname, format_name),
                scale_sweep=True,
                scale_coding=cb.SCALE_CODING_V1,
                encode_tier="balanced",
            )
            expected_parts.append(packed.reshape(1, ROWS, -1))
            expected_scales.append(fields["scales"].reshape(1, ROWS))
        slug = format_name.lower()
        prefix = f"{PACKED}.format_group_{slug}"
        assert torch.equal(
            emitted[prefix + ".cb_qweight"],
            torch.cat(expected_parts, dim=1).to(torch.uint8),
        )
        assert torch.equal(
            emitted[prefix + ".weight_scale"],
            torch.cat(expected_scales, dim=1).to(torch.float32),
        )
        down_weight = fixture["down_stack"][expert_id:expert_id + 1]
        down_cw = fixture["down_cw"][expert_id:expert_id + 1]
        down_packed, down_fields = cb.nvfp4_cb_pack(
            down_weight,
            rung,
            grid="fp8",
            mode="product",
            col_weights=down_cw,
            codebook=bundle.codebook_for(DOWN, format_name),
            scale_sweep=True,
            scale_coding=cb.SCALE_CODING_V1,
            encode_tier="balanced",
        )
        down_prefix = f"{PACKED_DOWN}.format_group_{slug}"
        assert torch.equal(
            emitted[down_prefix + ".cb_qweight"],
            down_packed.reshape(1, ROWS, -1).to(torch.uint8),
        )
        assert torch.equal(
            emitted[down_prefix + ".weight_scale"],
            down_fields["scales"].reshape(1, ROWS).to(torch.float32),
        )

    quant_config = json.loads((out / "quant_config.json").read_text())
    learned_groups = [
        group for group in quant_config["config_groups"].values()
        if group.get("format") in {FORMAT, FORMAT_32}
    ]
    expected_targets = {
        f"model.layers.0.mlp.experts.{projection}.format_group_{fmt.lower()}"
        for projection in ("gate_proj", "up_proj", "down_proj")
        for fmt in (FORMAT, FORMAT_32)
    }
    assert {
        group["targets"][0] for group in learned_groups
    } == expected_targets
    refs_by_target = {
        group["targets"][0]: tuple(group["scheme"]["codebook_ref"])
        for group in learned_groups
    }
    for format_name in (FORMAT, FORMAT_32):
        role_refs = {
            refs_by_target[
                "model.layers.0.mlp.experts."
                f"{projection}.format_group_{format_name.lower()}"
            ]
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        assert len(role_refs) == 3
    declaration = quant_config["per_expert_format_groups"]["layers"]["0"]
    assert {
        (entry["format_wire_id"], tuple(entry["expert_ids"]))
        for entry in declaration["w13"]
    } == {(FORMAT, (0,)), (FORMAT_32, (1,))}


def test_streaming_routed_books_refuse_an_fp8_ldlq_scope(
    tmp_path, monkeypatch
):
    import importlib

    stream = importlib.import_module(
        "prismaquant.export_nvfp4_cb_streaming"
    )
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        stream, "detect_profile", lambda *_args, **_kwargs: _PerExpertProfile()
    )
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_SCOPE", "all")
    context = cb_serialization_context_from_env()
    config_path = _streaming_layer_config(
        tmp_path / "ldlq-layer-config.json", fixture, context
    )
    with pytest.raises(ValueError, match="immutable no-LDLQ burn identity"):
        stream.export_nvfp4_cb_streaming(
            fixture["model_dir"],
            config_path,
            tmp_path / "ldlq-out",
            fixture["col_weights"],
            shared_codebook_spec={"source": "learned"},
            device="cpu",
        )
