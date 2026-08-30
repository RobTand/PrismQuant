"""Real-entry-point proofs for trellis fused and packed aggregation.

These tests license deletion of ``UNWIRED_LINKS`` #3 and #4.  They drive
``allocator.main()`` with real arguments and let the real solver choose the
assignment.  A thin spy observes the post-aggregation menu while delegating
unchanged to ``solve_with_promotion``; it never injects a candidate or an
assignment.

Expected bytes are computed independently from the wire descriptor.  In the
packed case the multiplier comes from the architecture's same physical export
contract: GLM emits one rank-2 wire per expert and projection, so gate_up is
``E * 2`` wires and down is ``E * 1`` wires.  Pricing either rank-3 parent as
one flattened wire would make this test fail at the exact campaign boundary.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections.abc import Mapping

import pytest

import prismaquant.allocator as alloc
from prismaquant import trellis_menu as tm
from prismaquant.allocator_candidates import (
    _FUSED_SIBLING_MARKER,
    _PACKED_GROUP_MARKER,
)
from prismaquant.trellis_footprint import trellis_tensor_payload_breakdown
from prismaquant.trellis_formats import (
    E4M3_FAMILY,
    LAYOUT_TIGHT_OFFSETS,
    get_trellis_family,
    native_code_value,
    parse_trellis_format_name,
)
from prismaquant.trellis_rate_surface import uniform_column_schedule


PROFILE = "trellis_research_sm121"
COST_MODE = "local"
CURRENCY = "weight-recon"
RUNG = "TCQ_E4M3_R1024"
_ANCHORS = [
    {"q256": 512, "dloss": 4.0e-3, "stderr": 0.0},
    {"q256": 1024, "dloss": 1.0e-3, "stderr": 0.0},
]


def _alphabet(rate: int) -> list[int]:
    spec = get_trellis_family(E4M3_FAMILY)
    required = 1 << (rate + 1)
    codes = [code for code in range(256) if code not in (0x7F, 0xFF)]
    if required > len(codes):
        codes.extend((0x00, 0x80))
        required = len(codes)
    codes.sort(key=lambda code: (native_code_value(spec, code), code))
    start = (len(codes) - required) // 2
    return list(codes[start:start + required])


_ALPHABETS = {rate: _alphabet(rate) for rate in range(1, 8)}


def _descriptor_bytes(shape: tuple[int, int], fmt: str = RUNG) -> int:
    """Compute one rank-2 wire's bytes without reading a Candidate."""

    parsed = parse_trellis_format_name(fmt)
    assert parsed is not None, fmt
    family, rate = parsed
    schedule = uniform_column_schedule(
        shape[1], rate, family=family,
    )
    used = sorted(
        code for code in set(schedule) if code < family.bypass_rate
    )
    payload = trellis_tensor_payload_breakdown(
        shape,
        family=family,
        body_rate_q256=rate,
        layout=LAYOUT_TIGHT_OFFSETS,
        schedule=schedule,
        alphabets={code: _ALPHABETS[code] for code in used},
    )
    return int(payload["total_bytes"])


def _write_inputs(
    tmp_path,
    *,
    model_type: str,
    architecture: str,
    stats: dict[str, dict],
    packed_parent_contracts: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[object, object, object, object]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": model_type,
        "architectures": [architecture],
        "tie_word_embeddings": False,
    }))

    probe = tmp_path / "probe.pkl"
    probe.write_bytes(pickle.dumps({
        "stats": stats,
        "meta": {"model": str(model_dir)},
    }))
    costs = tmp_path / "cost.pkl"
    costs.write_bytes(pickle.dumps({
        "costs": {
            name: {"BF16": {"predicted_dloss": 0.0}}
            for name in stats
        },
        "formats": ["BF16"],
        "provenance": {"cost_mode": COST_MODE},
    }))

    surface = tmp_path / "surface.json"
    surface.write_text(json.dumps({
        "schema": tm.TRELLIS_SURFACE_MANIFEST_SCHEMA,
        "cost_mode": COST_MODE,
        "currency": CURRENCY,
        "target_profile": PROFILE,
        "activation_contract": "W8A16",
        "layout": LAYOUT_TIGHT_OFFSETS,
        "rungs_per_unit": 2,
        "provenance": {
            "fixture": "synthetic measured parent-level anchors",
        },
        "anchors": {
            name: {
                "family": E4M3_FAMILY,
                "alphabets": {
                    str(rate): codes for rate, codes in _ALPHABETS.items()
                },
                "points": _ANCHORS,
                **(
                    {"packed_parent": dict(packed_parent_contracts[name])}
                    if packed_parent_contracts is not None
                    and name in packed_parent_contracts
                    else {}
                ),
            }
            for name in stats
        },
    }))
    return model_dir, probe, costs, surface


def _run_real_main(
    monkeypatch,
    tmp_path,
    *,
    model_type: str,
    architecture: str,
    stats: dict[str, dict],
    packed_parent_contracts: Mapping[str, Mapping[str, object]] | None = None,
):
    model_dir, probe, costs, surface = _write_inputs(
        tmp_path,
        model_type=model_type,
        architecture=architecture,
        stats=stats,
        packed_parent_contracts=packed_parent_contracts,
    )
    snapshots: list[dict[str, tuple]] = []
    real_solve = alloc.solve_with_promotion

    def observe_then_solve(
        solver_stats,
        candidates,
        target_bits,
        format_specs,
        format_rank,
        bit_precision,
        **kwargs,
    ):
        snapshots.append({
            name: tuple(menu) for name, menu in candidates.items()
        })
        return real_solve(
            solver_stats,
            candidates,
            target_bits,
            format_specs,
            format_rank,
            bit_precision,
            **kwargs,
        )

    monkeypatch.setenv(tm.TRELLIS_SURFACE_ENV, str(surface))
    monkeypatch.setattr(alloc, "solve_with_promotion", observe_then_solve)
    layer_config = tmp_path / "layer_config.json"
    monkeypatch.setattr(sys, "argv", [
        "allocator",
        "--probe", str(probe),
        "--costs", str(costs),
        "--model-override", str(model_dir),
        "--formats", "BF16",
        "--cost-mode", COST_MODE,
        "--target-profile", PROFILE,
        "--target-bits", "4.75",
        "--pareto-targets", "4.75",
        "--bit-precision", "0.01",
        "--layer-config", str(layer_config),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
    ])
    alloc.main()
    assert snapshots, "allocator.main() never called the real solver"
    return json.loads(layer_config.read_text()), snapshots


def _packed_parent_contract(
    *,
    source_shape: tuple[int, int, int],
    packed_param: str,
    projection_names: tuple[str, ...],
) -> dict[str, object]:
    """Declare what the synthetic anchors measured, independently of code."""

    experts, packed_rows, columns = source_shape
    assert packed_rows % len(projection_names) == 0
    return {
        "schema": tm.TRELLIS_PACKED_PARENT_ANCHOR_SCHEMA,
        "dloss_scope": tm.PACKED_PARENT_DLOSS_SCOPE,
        "wire_decomposition": tm.PACKED_WIRE_DECOMPOSITION,
        "model_profile": "glm5_next",
        "source_shape": list(source_shape),
        "packed_param": packed_param,
        "projection_names": list(projection_names),
        "projection_shape": [packed_rows // len(projection_names), columns],
        "wire_count": experts * len(projection_names),
    }


def _one_super_candidate(snapshots, marker: str):
    matches = []
    for snapshot in snapshots:
        for name, menu in snapshot.items():
            if marker not in name:
                continue
            matches.extend(candidate for candidate in menu if candidate.fmt == RUNG)
    assert matches, (
        f"the post-aggregation menu never offered {RUNG} on a {marker!r} "
        "super-item"
    )
    first = matches[0]
    assert all(candidate == first for candidate in matches)
    return first


def test_allocator_main_carries_trellis_rung_through_fused_aggregation(
        monkeypatch, tmp_path):
    names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    shape = (256, 256)
    stats = {
        name: {
            "h_trace": 1.0,
            "n_params": shape[0] * shape[1],
            "out_features": shape[0],
            "in_features": shape[1],
        }
        for name in names
    }

    payload, snapshots = _run_real_main(
        monkeypatch,
        tmp_path,
        model_type="qwen3",
        architecture="Qwen3ForCausalLM",
        stats=stats,
    )
    super_candidate = _one_super_candidate(
        snapshots, _FUSED_SIBLING_MARKER,
    )
    expected_bytes = len(names) * _descriptor_bytes(shape)
    expected_params = len(names) * shape[0] * shape[1]
    assert super_candidate.memory_bytes == expected_bytes
    assert super_candidate.bits_per_param == pytest.approx(
        8.0 * expected_bytes / expected_params,
    )
    assert super_candidate.predicted_dloss == pytest.approx(
        len(names) * 1.0e-3,
    )
    assert all(payload[name] == RUNG for name in names)


def test_allocator_main_carries_trellis_rung_through_packed_expert_aggregation(
        monkeypatch, tmp_path):
    gate_up = "model.language_model.layers.3.mlp.experts.gate_up_proj"
    down = "model.language_model.layers.3.mlp.experts.down_proj"
    gate_shape = (288, 4096, 4096)
    down_shape = (288, 4096, 2048)
    stats = {
        gate_up: {
            "h_trace": 1.0,
            "n_params": gate_shape[0] * gate_shape[1] * gate_shape[2],
            "out_features": gate_shape[1],
            "in_features": gate_shape[2],
            "num_experts": gate_shape[0],
            "_packed_experts_module": (
                "model.language_model.layers.3.mlp.experts"
            ),
            "_packed_param": "gate_up_proj",
        },
        down: {
            "h_trace": 1.0,
            "n_params": down_shape[0] * down_shape[1] * down_shape[2],
            "out_features": down_shape[1],
            "in_features": down_shape[2],
            "num_experts": down_shape[0],
            "_packed_experts_module": (
                "model.language_model.layers.3.mlp.experts"
            ),
            "_packed_param": "down_proj",
        },
    }

    payload, snapshots = _run_real_main(
        monkeypatch,
        tmp_path,
        model_type="glm5_next",
        architecture="Glm5NextForConditionalGeneration",
        stats=stats,
        packed_parent_contracts={
            gate_up: _packed_parent_contract(
                source_shape=gate_shape,
                packed_param="gate_up_proj",
                projection_names=("gate_proj", "up_proj"),
            ),
            down: _packed_parent_contract(
                source_shape=down_shape,
                packed_param="down_proj",
                projection_names=("down_proj",),
            ),
        },
    )
    super_candidate = _one_super_candidate(
        snapshots, _PACKED_GROUP_MARKER,
    )

    # GLM's architecture contract splits gate_up into gate/up and emits each
    # projection once per expert; down has one projection per expert.  These
    # counts are intentionally independent of the implementation under test.
    gate_wire_bytes = _descriptor_bytes(
        (gate_shape[1] // 2, gate_shape[2]),
    )
    down_wire_bytes = _descriptor_bytes(
        (down_shape[1], down_shape[2]),
    )
    assert gate_wire_bytes == 4_204_735
    assert down_wire_bytes == 4_211_871
    expected_gate = gate_shape[0] * 2 * gate_wire_bytes
    expected_down = down_shape[0] * down_wire_bytes
    expected_bytes = expected_gate + expected_down
    expected_params = (
        gate_shape[0] * gate_shape[1] * gate_shape[2]
        + down_shape[0] * down_shape[1] * down_shape[2]
    )
    assert expected_bytes == 3_634_946_208
    exact_bpw = 8.0 * expected_bytes / expected_params
    assert exact_bpw == pytest.approx(
        4.01221625,
    )
    # This is why the physical decomposition is an input, not bookkeeping.
    # Flattening each rank-3 parent to one rank-2 wire erases 862 headers and
    # alphabet directories and crosses the rival's 4.0117-bpw line.
    flattened_bytes = (
        _descriptor_bytes(
            (gate_shape[0] * gate_shape[1], gate_shape[2]),
        )
        + _descriptor_bytes(
            (down_shape[0] * down_shape[1], down_shape[2]),
        )
    )
    assert flattened_bytes == 3_633_319_262
    assert 8.0 * flattened_bytes / expected_params < 4.0117 < exact_bpw
    assert super_candidate.memory_bytes == expected_bytes
    assert super_candidate.bits_per_param == pytest.approx(
        8.0 * expected_bytes / expected_params,
    )
    assert super_candidate.predicted_dloss == pytest.approx(2.0e-3)
    # Golden over the sorted pair of physical parent-recipe digests.  This
    # kills a recipe that forgets shape, packed role, projection order/count,
    # per-wire identity, expert multiplicity, or exact bytes.
    assert super_candidate.serialized_identity == (
        '["7066f8718c479295b03d25deea2aa98cc48afe7f6106b601b6ae4f17be2257e1",'
        '"e10724b65ab41933b95a0f9a230dd77221440d30c921967b7f2db4823d388490"]'
    )
    assert all(payload[name] == RUNG for name in (gate_up, down))

    surface_stamp = payload["__prismaquant__"]["trellis_surface"]
    layouts = surface_stamp["packed_wire_layouts"]
    assert layouts[gate_up] == {
        "schema": tm.TRELLIS_PACKED_PARENT_ANCHOR_SCHEMA,
        "dloss_scope": tm.PACKED_PARENT_DLOSS_SCOPE,
        "wire_decomposition": tm.PACKED_WIRE_DECOMPOSITION,
        "model_profile": "glm5_next",
        "source_shape": [288, 4096, 4096],
        "packed_param": "gate_up_proj",
        "projection_names": ["gate_proj", "up_proj"],
        "projection_shape": [2048, 4096],
        "wire_count": 576,
        "experts": 288,
    }
    assert layouts[down]["wire_count"] == 288


def _small_legal_packed_gate_case():
    gate_up = "model.language_model.layers.3.mlp.experts.gate_up_proj"
    # A legal 256-column superblock and even gate/up split.  Keeping the path
    # legal matters: when a contract-check mutation is present, main must
    # reach pricing rather than fail later for an unrelated fixture defect.
    shape = (2, 512, 256)
    stats = {
        gate_up: {
            "h_trace": 1.0,
            "n_params": shape[0] * shape[1] * shape[2],
            "out_features": shape[1],
            "in_features": shape[2],
            "num_experts": shape[0],
            "_packed_experts_module": (
                "model.language_model.layers.3.mlp.experts"
            ),
            "_packed_param": "gate_up_proj",
        },
    }
    return gate_up, shape, stats


def test_allocator_main_refuses_packed_anchor_without_typed_parent_contract(
        monkeypatch, tmp_path):
    """Packed points with no declared measurement scope are unusable."""

    _gate_up, _shape, stats = _small_legal_packed_gate_case()
    with pytest.raises(
        tm.TrellisPackedExpertLayoutError,
        match=r"missing the typed 'packed_parent' contract",
    ):
        _run_real_main(
            monkeypatch,
            tmp_path,
            model_type="glm5_next",
            architecture="Glm5NextForConditionalGeneration",
            stats=stats,
        )


def test_allocator_main_refuses_packed_anchor_without_whole_parent_loss_scope(
        monkeypatch, tmp_path):
    """A parent loss is never inferred from an untyped/per-wire point."""

    gate_up, shape, stats = _small_legal_packed_gate_case()
    bad_contract = _packed_parent_contract(
        source_shape=shape,
        packed_param="gate_up_proj",
        projection_names=("gate_proj", "up_proj"),
    )
    bad_contract["dloss_scope"] = "per_wire"

    with pytest.raises(
        tm.TrellisPackedExpertLayoutError,
        match=r"dloss_scope.*whole_packed_parent.*per_wire",
    ):
        _run_real_main(
            monkeypatch,
            tmp_path,
            model_type="glm5_next",
            architecture="Glm5NextForConditionalGeneration",
            stats=stats,
            packed_parent_contracts={gate_up: bad_contract},
        )
