from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import check_format_applicability
from prismaquant.build_rtn_cache import kl_divergence
import prismaquant.kl_sensitivity_probe as ksp
from prismaquant.kl_sensitivity_probe import (
    FrontierPoint,
    LinearTarget,
    ProbeRow,
    UnitOption,
    choose_kneedle_point,
    _build_unit_options,
    _fused_assignment_violations,
    measure_frontier_points,
    _replay_cache_window_size,
    solve_multi_choice_frontier,
)
from prismaquant.iterate_perturbed_allocation import measure_assignment_kl
from prismaquant.model_profiles import Qwen3Profile
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.propagated_cost import resolve_kl_scope


def test_full_sequence_kl_equals_average_of_position_kls():
    teacher_logits = torch.tensor(
        [[[2.0, -1.0, 0.5], [0.25, 1.25, -0.5], [-1.0, 0.0, 2.0]]],
        dtype=torch.float32,
    )
    student_logits = torch.tensor(
        [[[1.5, -0.25, 0.0], [1.0, 0.25, -0.75], [-0.5, 1.5, 0.25]]],
        dtype=torch.float32,
    )
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    full_sequence = kl_divergence(student_logits, teacher_log_probs)
    per_position = torch.stack([
        kl_divergence(
            student_logits[:, idx:idx + 1, :],
            teacher_log_probs[:, idx:idx + 1, :],
        )
        for idx in range(student_logits.size(1))
    ]).mean()

    assert full_sequence.item() == pytest.approx(per_position.item(), abs=1e-12)


def test_explicit_kl_scope_wins_over_legacy_env(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_FULL_SEQUENCE_KL", "1")

    assert resolve_kl_scope(None) == "full_sequence"
    assert resolve_kl_scope("last_token") == "last_token"
    assert resolve_kl_scope("full_sequence") == "full_sequence"


def test_measure_assignment_kl_scope_controls_reduction(tmp_path, monkeypatch):
    class _Output:
        def __init__(self, logits):
            self.logits = logits

    class _KnownLogits(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = torch.nn.Parameter(logits, requires_grad=False)

        def forward(self, input_ids):
            return _Output(self.logits[: input_ids.size(0)])

    monkeypatch.setenv("PRISMAQUANT_FULL_SEQUENCE_KL", "1")
    monkeypatch.setenv("PRISMAQUANT_KL_CUDA_GRAPHS", "0")
    student_logits = torch.tensor(
        [[[1.5, -0.25, 0.0], [1.0, 0.25, -0.75], [-0.5, 1.5, 0.25]]],
        dtype=torch.float32,
    )
    teacher_logits = torch.tensor(
        [[[2.0, -1.0, 0.5], [0.25, 1.25, -0.5], [-1.0, 0.0, 2.0]]],
        dtype=torch.float32,
    )
    ref_log_probs = [F.log_softmax(teacher_logits, dim=-1)]
    calib_ids = torch.ones(1, 3, dtype=torch.long)
    model = _KnownLogits(student_logits)

    last = measure_assignment_kl(
        model,
        {},
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
        kl_scope="last_token",
    )
    full = measure_assignment_kl(
        model,
        {},
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
        kl_scope="full_sequence",
    )
    legacy_env = measure_assignment_kl(
        model,
        {},
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
        kl_scope=None,
    )

    expected_last = kl_divergence(
        student_logits[:, -1:, :],
        ref_log_probs[0][:, -1:, :],
    ).item()
    expected_full = kl_divergence(student_logits, ref_log_probs[0]).item()
    assert last == pytest.approx(expected_last)
    assert full == pytest.approx(expected_full)
    assert legacy_env == pytest.approx(expected_full)
    assert last != pytest.approx(full)


def test_format_applicability_positive_and_negative_cases():
    assert check_format_applicability(
        (128, 128),
        fr.get_format("NVFP4"),
        qname="model.layers.0.mlp.down_proj",
        source_kind="bf16",
        target_profile="research",
    ).legal

    group_bad = check_format_applicability(
        (128, 17),
        fr.get_format("NVFP4"),
        qname="model.layers.0.mlp.down_proj",
        source_kind="bf16",
        target_profile="research",
    )
    assert not group_bad.legal
    assert group_bad.reason == "group_divisibility"

    profile_bad = check_format_applicability(
        (128, 128),
        fr.get_format("MXFP4"),
        qname="model.layers.0.self_attn.q_proj",
        source_kind="bf16",
        target_profile="vllm_qwen3_5_packed_moe",
    )
    assert not profile_bad.legal
    assert profile_bad.reason == "profile_mismatch"

    source_bad = check_format_applicability(
        (256, 256),
        fr.get_format("FP8_SOURCE"),
        qname="model.layers.0.mlp.down_proj",
        source_kind="bf16",
        target_profile="research",
    )
    assert not source_bad.legal
    assert source_bad.reason == "source_dtype_mismatch"

    source_unknown = check_format_applicability(
        (256, 256),
        fr.get_format("FP8_SOURCE"),
        qname="model.layers.0.mlp.down_proj",
        source_kind=None,
        target_profile="research",
    )
    assert not source_unknown.legal
    assert source_unknown.reason == "source_dtype_mismatch"


def test_multi_choice_frontier_finds_non_greedy_knapsack_optimum():
    floor_assignment = {"a": "NVFP4", "b": "NVFP4", "c": "NVFP4"}
    options = {
        "a": [
            UnitOption("a", "NVFP4", ("a",), 100.0, 0.0, 0.0),
            UnitOption("a", "FP8_E4M3", ("a",), 110.0, 10.0, 60.0),
        ],
        "b": [
            UnitOption("b", "NVFP4", ("b",), 100.0, 0.0, 0.0),
            UnitOption("b", "FP8_E4M3", ("b",), 120.0, 20.0, 100.0),
        ],
        "c": [
            UnitOption("c", "NVFP4", ("c",), 100.0, 0.0, 0.0),
            UnitOption("c", "FP8_E4M3", ("c",), 130.0, 30.0, 120.0),
        ],
    }

    frontier = solve_multi_choice_frontier(
        options,
        floor_assignment=floor_assignment,
        floor_kl=1.0,
        budget_points=7,
        bit_precision_bits=1.0,
    )
    at_350 = max(
        (point for point in frontier if point.budget_bits <= 350.0),
        key=lambda point: point.gain,
    )

    assert at_350.gain == pytest.approx(220.0)
    assert at_350.assignment == {
        "a": "NVFP4",
        "b": "FP8_E4M3",
        "c": "FP8_E4M3",
    }
    assert choose_kneedle_point(frontier) >= 0


def test_kneedle_can_select_from_measured_frontier_gain():
    points = solve_multi_choice_frontier(
        {
            "a": [
                UnitOption("a", "NVFP4", ("a",), 100.0, 0.0, 0.0),
                UnitOption("a", "FP8_E4M3", ("a",), 120.0, 20.0, 8.0),
            ],
            "b": [
                UnitOption("b", "NVFP4", ("b",), 100.0, 0.0, 0.0),
                UnitOption("b", "FP8_E4M3", ("b",), 140.0, 40.0, 100.0),
            ],
        },
        floor_assignment={"a": "NVFP4", "b": "NVFP4"},
        floor_kl=1.0,
        budget_points=4,
        bit_precision_bits=1.0,
    )
    assert len(points) > 1
    measured = [
        replace(point, measured_kl=1.0 - float(idx), measured_gain=float(idx))
        for idx, point in enumerate(points)
    ]

    assert choose_kneedle_point(measured, use_measured=True) >= 0
    with pytest.raises(ValueError):
        choose_kneedle_point(points, use_measured=True)


def test_measure_frontier_points_reuses_floor_kl(monkeypatch, tmp_path):
    floor_assignment = {"a": "NVFP4", "b": "NVFP4"}
    promoted_assignment = {"a": "BF16", "b": "NVFP4"}
    frontier = [
        FrontierPoint(
            budget_bits=100.0,
            bits_total=100.0,
            bits_delta=0.0,
            gain=0.0,
            predicted_kl=0.75,
            unit_assignment=dict(floor_assignment),
            assignment=dict(floor_assignment),
            promotion_count=0,
        ),
        FrontierPoint(
            budget_bits=120.0,
            bits_total=120.0,
            bits_delta=20.0,
            gain=0.25,
            predicted_kl=0.5,
            unit_assignment=dict(promoted_assignment),
            assignment=dict(promoted_assignment),
            promotion_count=1,
        ),
    ]
    calls = []

    def _measure(_model, assignment, *_args, **_kwargs):
        calls.append(dict(assignment))
        return 0.4

    monkeypatch.setattr(ksp, "measure_assignment_kl", _measure)

    measured = measure_frontier_points(
        torch.nn.Linear(1, 1),
        frontier,
        torch.ones(1, 1, dtype=torch.long),
        [torch.zeros(1, 1, 1)],
        floor_kl=0.75,
        floor_assignment=floor_assignment,
        work_root=tmp_path,
        profile=None,
        kl_scope="last_token",
    )

    assert calls == [promoted_assignment]
    assert measured[0].measured_kl == pytest.approx(0.75)
    assert measured[0].measured_gain == pytest.approx(0.0)
    assert measured[1].measured_kl == pytest.approx(0.4)
    assert measured[1].measured_gain == pytest.approx(0.35)


def test_seed_assignment_loader_normalizes_current_oracle_candidate(tmp_path):
    class _Profile:
        def fused_sibling_group(self, qname):
            if qname.endswith(("gate_proj", "up_proj")):
                return "model.layers.0.mlp.gate_up_proj"
            return None

    targets = [
        LinearTarget("model.layers.0.mlp.gate_proj", (128, 128), 16384),
        LinearTarget("model.layers.0.mlp.up_proj", (128, 128), 16384),
        LinearTarget("model.layers.0.mlp.down_proj", (128, 128), 16384),
        LinearTarget("model.layers.0.self_attn.o_proj", (128, 128), 16384),
        LinearTarget("lm_head", (128, 128), 16384, pinned=True),
    ]
    floor_assignment = {
        target.qname: ("BF16" if target.pinned else "NVFP4")
        for target in targets
    }
    seed_path = tmp_path / "historical_assignment.json"
    seed_path.write_text(json.dumps({
        "label": "old_ship_candidate",
        "assignment": {
            "model.layers.0.mlp.gate_proj": "BF16",
            "model.layers.0.mlp.down_proj": "MXFP8",
            "model.layers.0.self_attn.o_proj": "NOPE",
            "lm_head": "NVFP4",
            "not.in.this.model": "BF16",
        },
    }))

    points, diagnostics = ksp._load_seed_assignment_points(
        [str(seed_path)],
        floor_assignment=floor_assignment,
        floor_kl=0.25,
        targets=targets,
        profile=_Profile(),
        requested_formats=["NVFP4", "MXFP8_E4M3", "BF16"],
        source_manifest={target.qname: "bf16" for target in targets},
        target_profile="research",
    )

    assert len(points) == 1
    point = points[0]
    diag = diagnostics[0]
    assert point.source == "seed_assignment"
    assert point.label == "old_ship_candidate"
    assert point.predicted_kl == pytest.approx(0.25)
    assert point.assignment["model.layers.0.mlp.gate_proj"] == "BF16"
    assert point.assignment["model.layers.0.mlp.up_proj"] == "BF16"
    assert point.assignment["model.layers.0.mlp.down_proj"] == "MXFP8_E4M3"
    assert point.assignment["model.layers.0.self_attn.o_proj"] == "NVFP4"
    assert point.assignment["lm_head"] == "BF16"
    assert diag["unknown_entries"] == 1
    assert diag["unknown_sample"] == ["not.in.this.model"]
    assert diag["invalid_formats"][0]["format"] == "NOPE"
    assert diag["pinned_conflicts"][0]["qname"] == "lm_head"
    assert diag["included"] is True


def test_seed_assignment_points_dedupe_against_frontier():
    floor_assignment = {"a": "NVFP4", "b": "NVFP4"}
    frontier = [
        FrontierPoint(
            budget_bits=10.0,
            bits_total=10.0,
            bits_delta=0.0,
            gain=0.0,
            predicted_kl=1.0,
            unit_assignment=dict(floor_assignment),
            assignment=dict(floor_assignment),
            promotion_count=0,
        )
    ]
    seed = replace(
        frontier[0],
        source="seed_assignment",
        label="duplicate_floor",
    )
    combined, diagnostics = ksp._append_unique_frontier_points(
        frontier,
        [seed],
        [{"assignment_hash": ksp._assignment_digest(seed.assignment)}],
    )

    assert combined == frontier
    assert diagnostics[0]["included"] is False
    assert diagnostics[0]["duplicate_of_frontier_index"] == 0


def test_qwen3_profile_has_vllm_packed_module_fallback_without_vllm():
    profile = Qwen3Profile()
    profile._vllm_cls = None
    profile._vllm_cls_loaded = True
    profile._fused_matcher = None

    assert (
        profile.fused_sibling_group("model.layers.0.self_attn.q_proj")
        == "model.layers.0.self_attn.qkv_proj"
    )
    assert (
        profile.fused_sibling_group("model.layers.0.self_attn.k_proj")
        == "model.layers.0.self_attn.qkv_proj"
    )
    assert (
        profile.fused_sibling_group("model.layers.0.mlp.gate_proj")
        == "model.layers.0.mlp.gate_up_proj"
    )
    assert profile.fused_sibling_group("model.layers.0.mlp.down_proj") is None


def test_probe_frontier_groups_vllm_packed_modules_into_coherent_assignment():
    class _Profile:
        def fused_sibling_group(self, qname):
            if qname.endswith(("gate_proj", "up_proj")):
                return "model.layers.0.mlp.gate_up_proj"
            return None

    targets = [
        LinearTarget("model.layers.0.mlp.gate_proj", (4, 4), 16),
        LinearTarget("model.layers.0.mlp.up_proj", (4, 4), 16),
        LinearTarget("model.layers.0.mlp.down_proj", (4, 4), 16),
    ]
    floor_assignment = {target.qname: "NVFP4" for target in targets}
    rows = [
        ProbeRow(
            qname=target.qname,
            format=fmt,
            shape=target.shape,
            bits_baseline=100.0,
            bits_format=bits,
            bits_delta=bits - 100.0,
            candidate_kl=1.0 - gain,
            sensitivity=gain,
        )
        for target in targets
        for fmt, bits, gain in [
            ("NVFP4", 100.0, 0.0),
            ("MXFP8_E4M3", 140.0, 1.0),
            ("BF16", 200.0, 2.0),
        ]
    ]

    options, unit_for_qname, missing = _build_unit_options(
        rows,
        targets,
        floor_format="NVFP4",
        floor_assignment=floor_assignment,
        profile=_Profile(),
    )

    assert missing == {}
    assert unit_for_qname["model.layers.0.mlp.gate_proj"] == (
        "model.layers.0.mlp.gate_up_proj"
    )
    assert options["model.layers.0.mlp.gate_up_proj"][1].members == (
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    )

    frontier = solve_multi_choice_frontier(
        options,
        floor_assignment=floor_assignment,
        floor_kl=10.0,
        budget_points=4,
        bit_precision_bits=1.0,
    )
    assert frontier
    for point in frontier:
        assert _fused_assignment_violations(
            point.assignment, targets, _Profile()
        ) == {}


def test_replay_cache_auto_window_caps_effective_lane_batch():
    class _Config:
        hidden_size = 16

    class _Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Config()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in range(4)]
            )

    window = _replay_cache_window_size(
        _Toy(),
        torch.ones(64, 32, dtype=torch.long),
        dtype=torch.float32,
        window_arg="auto",
        max_cache_gb=128.0,
        max_lanes_per_batch=4,
        max_effective_batch=16,
    )

    assert window == 4


def test_candidate_replay_checkpoint_round_trips_and_rejects_mismatch(tmp_path):
    path = tmp_path / "candidate_overrides_tail_replay_checkpoint.json"
    calib_ids = torch.arange(8, dtype=torch.long).reshape(2, 4)
    overrides = [
        {"model.layers.0.mlp.down_proj": "BF16"},
        {"model.layers.1.mlp.down_proj": "MXFP8_E4M3"},
    ]
    signature = ksp._candidate_replay_checkpoint_signature(
        floor_assignment={
            "model.layers.0.mlp.down_proj": "NVFP4",
            "model.layers.1.mlp.down_proj": "NVFP4",
        },
        ordered_overrides=overrides,
        calib_ids=calib_ids,
        kl_scope="last_token",
        include_activation_quant=False,
        max_lanes_per_batch=4,
        replay_cache_window="auto",
        replay_cache_max_gb=8.0,
        replay_cache_max_effective_batch=16,
    )
    windows = {
        0: {"start": 0, "end": 1, "rows": 1, "values": [0.1, 0.2]},
        1: {"start": 1, "end": 2, "rows": 1, "values": [0.3, 0.4]},
    }

    ksp._write_candidate_replay_checkpoint(
        path,
        signature=signature,
        total_windows=2,
        windows=windows,
    )

    loaded = ksp._load_candidate_replay_checkpoint(
        path,
        signature=signature,
        expected_candidates=2,
    )
    assert loaded == windows
    assert ksp._load_candidate_replay_checkpoint(
        path,
        signature="different",
        expected_candidates=2,
    ) == {}
    assert ksp._load_candidate_replay_checkpoint(
        path,
        signature=signature,
        expected_candidates=3,
    ) == {}


def test_adaptive_group_candidate_search_splits_only_improving_groups(
    monkeypatch,
    tmp_path,
):
    qnames = [
        f"model.layers.{idx}.mlp.down_proj"
        for idx in range(8)
    ]
    qname_to_idx = {qname: idx for idx, qname in enumerate(qnames)}
    targets = [
        LinearTarget(qname, (4, 4), 16)
        for qname in qnames
    ]
    overrides = [
        {target.qname: "BF16"}
        for target in targets
    ]
    candidate_meta = [
        (target.qname, (target.qname,), (target,), "BF16", 10.0, 20.0)
        for target in targets
    ]
    calls: list[list[tuple[int, ...]]] = []

    def _fake_measure(_model, _floor_assignment, candidate_overrides, *_args, **_kwargs):
        call_groups: list[tuple[int, ...]] = []
        values: list[float] = []
        for override in candidate_overrides:
            indices = tuple(sorted(qname_to_idx[qname] for qname in override))
            call_groups.append(indices)
            if len(indices) == 1:
                gain = {0: 0.10, 1: 0.03}.get(indices[0], -0.02)
            elif indices == (0, 1, 2, 3):
                gain = 0.40
            elif indices == (0, 1):
                gain = 0.20
            else:
                gain = -0.05
            values.append(1.0 - gain)
        calls.append(call_groups)
        return values

    monkeypatch.setattr(ksp, "measure_candidate_overrides", _fake_measure)

    result = ksp._adaptive_group_candidate_kls(
        torch.nn.Linear(1, 1),
        {target.qname: "NVFP4" for target in targets},
        overrides,
        candidate_meta,
        torch.ones(1, 1, dtype=torch.long),
        [torch.zeros(1, 1, 1)],
        floor_kl=1.0,
        work_root=tmp_path,
        profile=None,
        kl_scope="last_token",
        max_lanes_per_batch=4,
        calib_microbatch_size=1,
        include_activation_quant=False,
        use_cuda_graphs=False,
        use_tail_replay=False,
        replay_cache_window="auto",
        replay_cache_max_gb=1.0,
        replay_cache_max_effective_batch=4,
        dtype=torch.float32,
        group_size=4,
        min_group_gain=0.0,
        max_exact_candidates=0,
    )

    assert calls == [
        [(0, 1, 2, 3), (4, 5, 6, 7)],
        [(0, 1), (2, 3)],
        [(0,), (1,)],
    ]
    assert set(result.candidate_kls) == {0, 1}
    assert result.candidate_kls[0] == pytest.approx(0.90)
    assert result.candidate_kls[1] == pytest.approx(0.97)
    assert result.pruned_indices == (2, 3, 4, 5, 6, 7)
    assert result.diagnostics["measured_groups"] == 6
    assert result.diagnostics["measured_candidates"] == 2


def test_adaptive_group_candidate_search_can_split_before_pruning(
    monkeypatch,
    tmp_path,
):
    qnames = [
        f"model.layers.{idx}.mlp.down_proj"
        for idx in range(4)
    ]
    qname_to_idx = {qname: idx for idx, qname in enumerate(qnames)}
    targets = [
        LinearTarget(qname, (4, 4), 16)
        for qname in qnames
    ]
    overrides = [
        {target.qname: "BF16"}
        for target in targets
    ]
    candidate_meta = [
        (target.qname, (target.qname,), (target,), "BF16", 10.0, 20.0)
        for target in targets
    ]
    calls: list[list[tuple[int, ...]]] = []

    def _fake_measure(_model, _floor_assignment, candidate_overrides, *_args, **_kwargs):
        call_groups: list[tuple[int, ...]] = []
        values: list[float] = []
        for override in candidate_overrides:
            indices = tuple(sorted(qname_to_idx[qname] for qname in override))
            call_groups.append(indices)
            gain = {
                (0, 1, 2, 3): -0.20,
                (0, 1): 0.15,
                (2, 3): -0.10,
                (0,): 0.10,
                (1,): 0.04,
            }.get(indices, -0.01)
            values.append(1.0 - gain)
        calls.append(call_groups)
        return values

    monkeypatch.setattr(ksp, "measure_candidate_overrides", _fake_measure)

    result = ksp._adaptive_group_candidate_kls(
        torch.nn.Linear(1, 1),
        {target.qname: "NVFP4" for target in targets},
        overrides,
        candidate_meta,
        torch.ones(1, 1, dtype=torch.long),
        [torch.zeros(1, 1, 1)],
        floor_kl=1.0,
        work_root=tmp_path,
        profile=None,
        kl_scope="last_token",
        max_lanes_per_batch=4,
        calib_microbatch_size=1,
        include_activation_quant=False,
        use_cuda_graphs=False,
        use_tail_replay=False,
        replay_cache_window="auto",
        replay_cache_max_gb=1.0,
        replay_cache_max_effective_batch=4,
        dtype=torch.float32,
        group_size=4,
        min_group_gain=0.0,
        max_exact_candidates=0,
        prune_after_round=2,
    )

    assert calls == [
        [(0, 1, 2, 3)],
        [(0, 1), (2, 3)],
        [(0,), (1,)],
    ]
    assert set(result.candidate_kls) == {0, 1}
    assert result.pruned_indices == (2, 3)
    assert result.diagnostics["rounds"][0]["split_groups"] == 2


def test_adaptive_group_candidate_search_fails_open_when_all_groups_prune(
    monkeypatch,
    tmp_path,
):
    qnames = [
        f"model.layers.{idx}.mlp.down_proj"
        for idx in range(4)
    ]
    qname_to_idx = {qname: idx for idx, qname in enumerate(qnames)}
    targets = [
        LinearTarget(qname, (4, 4), 16)
        for qname in qnames
    ]
    overrides = [
        {target.qname: "BF16"}
        for target in targets
    ]
    candidate_meta = [
        (target.qname, (target.qname,), (target,), "BF16", 10.0, 20.0)
        for target in targets
    ]
    calls: list[list[tuple[int, ...]]] = []

    def _fake_measure(_model, _floor_assignment, candidate_overrides, *_args, **_kwargs):
        call_groups: list[tuple[int, ...]] = []
        values: list[float] = []
        for override in candidate_overrides:
            indices = tuple(sorted(qname_to_idx[qname] for qname in override))
            call_groups.append(indices)
            gain = {
                (0, 1, 2, 3): -0.30,
                (0, 1): -0.01,
                (2, 3): -0.20,
                (0,): -0.03,
                (1,): -0.04,
            }.get(indices, -0.10)
            values.append(1.0 - gain)
        calls.append(call_groups)
        return values

    monkeypatch.setattr(ksp, "measure_candidate_overrides", _fake_measure)

    result = ksp._adaptive_group_candidate_kls(
        torch.nn.Linear(1, 1),
        {target.qname: "NVFP4" for target in targets},
        overrides,
        candidate_meta,
        torch.ones(1, 1, dtype=torch.long),
        [torch.zeros(1, 1, 1)],
        floor_kl=1.0,
        work_root=tmp_path,
        profile=None,
        kl_scope="last_token",
        max_lanes_per_batch=4,
        calib_microbatch_size=1,
        include_activation_quant=False,
        use_cuda_graphs=False,
        use_tail_replay=False,
        replay_cache_window="auto",
        replay_cache_max_gb=1.0,
        replay_cache_max_effective_batch=4,
        dtype=torch.float32,
        group_size=4,
        min_group_gain=0.0,
        max_exact_candidates=2,
        prune_after_round=2,
        fail_open_pruning=True,
    )

    assert calls == [
        [(0, 1, 2, 3)],
        [(0, 1), (2, 3)],
        [(0,), (1,)],
    ]
    assert set(result.candidate_kls) == {0, 1}
    assert result.pruned_indices == (2, 3)
    assert result.diagnostics["rounds"][1]["fail_open_groups"] == 1
    assert result.diagnostics["rounds"][1]["fail_open_candidates"] == 2


def test_production_cache_metadata_validates_identity_and_entries():
    args = SimpleNamespace(
        model="/tmp/qwen",
        target_profile="qwen3",
        calib_split="train",
        calib_seed=42,
        production_cache_levers="gptq,scale_sweep",
        production_cache_max_act_rows=512,
    )
    calib_ids = torch.arange(8, dtype=torch.long).reshape(2, 4)
    expected = ksp._production_cache_expected_metadata(
        args,
        calib_ids,
        ["model.layers.0.mlp.down_proj"],
        ["NVFP4"],
        {"model.layers.0.mlp.down_proj": "bf16"},
    )
    cache = ProductionWeightCache(
        weights={
            ("model.layers.0.mlp.down_proj", "NVFP4"): torch.zeros(
                (2, 2), dtype=torch.bfloat16
            )
        },
        levers={"gptq": True, "scale_sweep": True},
    )

    metadata = ksp._attach_production_cache_metadata(cache, expected)
    status = ksp._validate_production_cache_metadata(cache, expected)

    assert metadata["identity_sha256"] == expected["identity_sha256"]
    assert status["validated"] is True
    assert status["status"] == "validated"

    changed_expected = ksp._production_cache_expected_metadata(
        args,
        calib_ids,
        ["model.layers.0.mlp.down_proj"],
        ["NVFP4"],
        {"model.layers.0.mlp.down_proj": "fp8"},
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        ksp._validate_production_cache_metadata(cache, changed_expected)

    legacy = ProductionWeightCache(
        weights={
            ("model.layers.0.mlp.down_proj", "NVFP4"): torch.zeros(
                (2, 2), dtype=torch.bfloat16
            )
        },
        levers={"gptq": True, "scale_sweep": True},
    )
    legacy_status = ksp._validate_production_cache_metadata(legacy, expected)
    assert legacy_status["status"] == "legacy_missing"
    assert legacy_status["validated"] is False


def test_kl_sensitivity_probe_help_parses():
    result = subprocess.run(
        [sys.executable, "-m", "prismaquant.kl_sensitivity_probe", "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "--kl-scope" in result.stdout


@pytest.mark.skipif(
    os.environ.get("PRISMAQUANT_RUN_QWEN_SMOKE") != "1",
    reason="set PRISMAQUANT_RUN_QWEN_SMOKE=1 for local model smoke",
)
def test_kl_sensitivity_probe_qwen_smoke(tmp_path):
    model = Path("/home/rob/.cache/huggingface/Qwen3-0.6B")
    if not model.exists():
        pytest.skip(f"{model} not present")
    output = tmp_path / "probe.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.kl_sensitivity_probe",
            "--model",
            str(model),
            "--output",
            str(output),
            "--work-root",
            str(tmp_path),
            "--floor-format",
            "NVFP4",
            "--formats",
            "registry",
            "--pin",
            "lm_head",
            "--calib-split",
            "train",
            "--n-calib-samples",
            "1",
            "--calib-seqlen",
            "32",
            "--calib-seed",
            "42",
            "--kl-scope",
            "last_token",
            "--max-lanes-per-batch",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["schema"] == "prismaquant.kl_sensitivity_probe.v1"
    assert payload["rows"]
    assert payload["floor"]["kl"] >= 0.0
