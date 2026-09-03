from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import math

import pytest
import torch

from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    assignment_serialization_sha256,
    cb_serialization_context_stamp,
)
from prismaquant.production_weight_cache import (
    bind_cb_render_identity_source_weights,
    build_production_cache_cb_render_identity,
)
from prismaquant.select_validated_frontier import (
    DEFAULT_TAIL_ETA,
    DEFAULT_TAIL_VETO,
    _frontier_from_rows,
    _kneedle_convex_decreasing,
    _log_error_values,
    _saturation_pick,
    leave_one_out_kneedle_diagnostic,
    measured_frontier,
    measured_rows,
    practical_knee,
    select_frontier_point,
    spearman_rank_correlation,
    tail_eta_auto,
    tail_veto_inert_reason,
    worst_rank_inversion,
)


def _production_cb_stamp(formats=("NVFP4_CB_K16",)):
    return cb_serialization_context_stamp(
        CBSerializationContext.production(), formats=formats
    )


def _canonical_payload_sha256(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _complete_cb_render_identity(formats_by_qname):
    context = CBSerializationContext.production()
    qnames = sorted(formats_by_qname)
    col_weights = {
        qname: torch.linspace(0.1, 1.0, 256)
        for qname in qnames
    }
    identity = build_production_cache_cb_render_identity(
        formats_by_qname,
        cb_serialization_context=context,
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    assert identity is not None
    return bind_cb_render_identity_source_weights(
        identity,
        {
            qname: torch.full((2, 256), index, dtype=torch.float32)
            for index, qname in enumerate(qnames, start=1)
        },
    )


def _sat_results(stderr):
    # flat tail (6.0..8.0 within noise of asymptote), decreasing before it
    rows = [(4.5, 0.10), (5.0, 0.06), (6.0, 0.030), (7.0, 0.029), (8.0, 0.028)]
    out = []
    for bpp, kl in rows:
        r = {"label": f"a{bpp}", "path": f"/x/a{bpp}.json", "bpp": bpp,
             "last_token_kl": kl, "format_counts": {}}
        if stderr is not None:
            r["kl_stderr"] = stderr
        out.append(r)
    return out


def test_saturation_mode_picks_bstar_with_real_stderr():
    sel, frontier = select_frontier_point(
        _sat_results(3e-3), mode="saturation", sat_z=2.0)
    assert sel["bpp"] == 6.0   # 6/7/8 indistinguishable within the band -> B*=6
    idx, sat = _saturation_pick(frontier, 2.0)
    assert sat["no_noise_floor"] is False
    assert frontier[idx]["bpp"] == 6.0


def test_saturation_mode_zero_stderr_flags_no_noise_floor():
    sel, frontier = select_frontier_point(
        _sat_results(0.0), mode="saturation", sat_z=2.0)
    assert sel["bpp"] == 8.0   # band collapses -> densest asymptote (most bits)
    _idx, sat = _saturation_pick(frontier, 2.0)
    assert sat["no_noise_floor"] is True


def test_saturation_mode_missing_stderr_key_is_no_noise_floor():
    # rows entirely lacking kl_stderr must not KeyError; treated as 0 stderr.
    sel, frontier = select_frontier_point(
        _sat_results(None), mode="saturation", sat_z=2.0)
    _idx, sat = _saturation_pick(frontier, 2.0)
    assert sat["no_noise_floor"] is True
    assert sel["bpp"] == 8.0


def test_saturation_single_point_frontier_does_not_crash():
    res = [{"label": "only", "path": "/x/only.json", "bpp": 6.0,
            "last_token_kl": 0.03, "kl_stderr": 1e-3}]
    sel, frontier = select_frontier_point(res, mode="saturation", sat_z=2.0)
    assert sel["bpp"] == 6.0 and len(frontier) == 1


def test_measured_frontier_drops_dominated_points():
    results = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "last_token_kl": 0.20},
        {"label": "b", "path": "b.json", "bpp": 4.6, "last_token_kl": 0.30},
        {"label": "c", "path": "c.json", "bpp": 5.0, "last_token_kl": 0.10},
        {"label": "d", "path": "d.json", "bpp": 5.5, "last_token_kl": 0.09},
    ]

    frontier = measured_frontier(results)

    assert [row["label"] for row in frontier] == ["a", "c", "d"]


def test_measured_rows_keep_dominated_points_for_diagnostics():
    results = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "last_token_kl": 0.10},
        {"label": "b", "path": "b.json", "bpp": 4.6, "last_token_kl": 0.30},
        {"label": "c", "path": "c.json", "bpp": 5.0, "last_token_kl": 0.05},
    ]

    assert [row["label"] for row in measured_rows(results)] == ["a", "b", "c"]
    assert [row["label"] for row in measured_frontier(results)] == ["a", "c"]


def test_select_frontier_best_kl():
    results = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "last_token_kl": 0.20},
        {"label": "b", "path": "b.json", "bpp": 5.0, "last_token_kl": 0.10},
        {"label": "c", "path": "c.json", "bpp": 5.5, "last_token_kl": 0.11},
    ]

    selected, frontier = select_frontier_point(results, mode="best-kl")

    assert selected["label"] == "b"
    assert [row["label"] for row in frontier] == ["a", "b"]


def test_measured_frontier_can_use_ucb_metric():
    results = [
        {
            "label": "a",
            "path": "a.json",
            "bpp": 4.5,
            "last_token_kl": 0.10,
            "kl_ucb": 0.30,
        },
        {
            "label": "b",
            "path": "b.json",
            "bpp": 5.0,
            "last_token_kl": 0.12,
            "kl_ucb": 0.20,
        },
    ]

    frontier = measured_frontier(results, metric="ucb")

    assert [row["label"] for row in frontier] == ["a", "b"]
    assert frontier[0]["kl"] == 0.30
    assert frontier[1]["kl"] == 0.20


def test_practical_knee_picks_lowest_bpp_within_tolerance():
    frontier = [
        {"label": "a", "path": "a.json", "bpp": 5.0, "kl": 0.101},
        {"label": "b", "path": "b.json", "bpp": 5.5, "kl": 0.100},
        {"label": "c", "path": "c.json", "bpp": 6.0, "kl": 0.090},
    ]

    selected = practical_knee(frontier, rel_eps=0.02)

    assert selected["label"] == "c"
    selected = practical_knee(frontier, rel_eps=0.13)
    assert selected["label"] == "a"


def test_select_frontier_reports_rank_and_leave_one_out_helpers():
    # kl_stderr >> any possible LOO shift: the stability tolerance derives
    # from the knee's measured repeat stderr, so the pick reads stable.
    # (kl_noise_floor must stay consistent with the eta used to build the
    # frontier — a floor larger than the KL deltas would collapse the
    # rebuilt leave-one-out envelopes to a single point.)
    frontier = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "kl": 0.30,
         "surrogate_loss": 3.0, "kl_stderr": 10.0},
        {"label": "b", "path": "b.json", "bpp": 5.0, "kl": 0.20,
         "surrogate_loss": 2.0, "kl_stderr": 10.0},
        {"label": "c", "path": "c.json", "bpp": 5.5, "kl": 0.10,
         "surrogate_loss": 1.0, "kl_stderr": 10.0},
        {"label": "d", "path": "d.json", "bpp": 6.0, "kl": 0.09,
         "surrogate_loss": 0.5, "kl_stderr": 10.0},
    ]

    assert spearman_rank_correlation(frontier) > 0.9
    diagnostic = leave_one_out_kneedle_diagnostic(
        frontier,
        frontier[1],
        tolerance_bpp=10.0,
    )
    assert diagnostic["enabled"]
    assert diagnostic["stable"]
    assert diagnostic["stability_tolerance_source"] == "repeat_stderr"


def test_measured_frontier_extracts_surrogate_from_nested_mse():
    # Real validate_assignments_kl rows carry the surrogate nested as
    # mse.predicted_dloss_sum, NOT a top-level surrogate_loss. This is the data
    # path that previously left surrogate_spearman silently None on every run.
    results = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "last_token_kl": 0.30,
         "mse": {"predicted_dloss_sum": 3.0}},
        {"label": "b", "path": "b.json", "bpp": 5.0, "last_token_kl": 0.20,
         "mse": {"predicted_dloss_sum": 2.0}},
        {"label": "c", "path": "c.json", "bpp": 5.5, "last_token_kl": 0.10,
         "mse": {"predicted_dloss_sum": 1.0}},
        {"label": "d", "path": "d.json", "bpp": 6.0, "last_token_kl": 0.09,
         "mse": {"predicted_dloss_sum": 0.5}},
    ]
    frontier = measured_frontier(results)
    for row in frontier:
        assert row["surrogate_loss"] is not None
    corr = spearman_rank_correlation(frontier)
    assert corr is not None
    assert corr > 0.9


def test_measured_frontier_top_level_surrogate_loss_takes_precedence():
    # Backward compat: an explicit top-level surrogate_loss still wins.
    results = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "last_token_kl": 0.30,
         "surrogate_loss": 9.0, "mse": {"predicted_dloss_sum": 3.0}},
    ]
    frontier = measured_frontier(results)
    assert frontier[0]["surrogate_loss"] == 9.0


def test_worst_rank_inversion_detects_mispredicted_pair():
    # 'a' is predicted best (lowest surrogate) but measured worst (highest KL).
    frontier = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "kl": 0.30, "surrogate_loss": 1.0},
        {"label": "b", "path": "b.json", "bpp": 5.0, "kl": 0.20, "surrogate_loss": 2.0},
        {"label": "c", "path": "c.json", "bpp": 5.5, "kl": 0.10, "surrogate_loss": 3.0},
    ]
    inv = worst_rank_inversion(frontier)
    assert inv is not None
    # 'a' (lowest surrogate) is the predicted-best of the worst inverted pair.
    assert inv["predicted_best_label"] == "a"
    assert inv["predicted_worse_label"] == "c"
    assert inv["rank_gap"] > 0.0
    assert "measured KL was worse" in inv["verdict"]


def test_worst_rank_inversion_none_when_concordant():
    # Perfectly concordant surrogate/KL ordering -> no inversion.
    frontier = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "kl": 0.30, "surrogate_loss": 3.0},
        {"label": "b", "path": "b.json", "bpp": 5.0, "kl": 0.20, "surrogate_loss": 2.0},
        {"label": "c", "path": "c.json", "bpp": 5.5, "kl": 0.10, "surrogate_loss": 1.0},
    ]
    assert worst_rank_inversion(frontier) is None


def test_worst_rank_inversion_none_when_too_few_pairs():
    frontier = [
        {"label": "a", "path": "a.json", "bpp": 4.5, "kl": 0.30, "surrogate_loss": 1.0},
        {"label": "b", "path": "b.json", "bpp": 5.0, "kl": 0.20, "surrogate_loss": 2.0},
    ]
    assert worst_rank_inversion(frontier) is None


def test_select_validated_frontier_cli_writes_layer_config(tmp_path):
    assignment_path = tmp_path / "candidate.json"
    assignment_path.write_text(json.dumps({
        "schema": "prismaquant.allocator.pareto_assignment.v1",
        "assignment": {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.mlp.down_proj": "MXFP8_E4M3",
            "model.layers.1.mlp.down_proj": "BF16",
        },
    }))
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({
        "results": [{
            "label": "candidate",
            "path": str(assignment_path),
            "bpp": 5.0,
            "last_token_kl": 0.01,
            "format_counts": {"NVFP4": 1, "MXFP8_E4M3": 1, "BF16": 1},
        }],
    }))
    layer_config = tmp_path / "layer_config.json"
    layer_config.write_text(json.dumps({
        "__prismaquant__": {
            "cb_serialized_payload": _production_cb_stamp(),
        },
    }))
    assignment_out = tmp_path / "selected_assignment.json"
    summary = tmp_path / "selection.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.select_validated_frontier",
            "--validation-json",
            str(validation_path),
            "--mode",
            "practical-knee",
            "--output-layer-config",
            str(layer_config),
            "--output-assignment",
            str(assignment_out),
            "--output-summary",
            str(summary),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    payload = json.loads(layer_config.read_text())
    assert set(payload) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.down_proj",
    }
    assert payload["model.layers.0.self_attn.q_proj"]["data_type"] == "nv_fp"
    assert payload["model.layers.0.mlp.down_proj"]["data_type"] == "mx_fp"
    assert payload["model.layers.1.mlp.down_proj"]["data_type"] == "float"

    selected = json.loads(summary.read_text())["selected"]
    assert selected["label"] == "candidate"


def test_selector_preserves_cb_global_and_per_layer_serialization_identity(
    tmp_path,
):
    qname = "model.layers.0.self_attn.q_proj"
    assignment = {qname: "NVFP4_CB_K16"}
    render_identity = _complete_cb_render_identity(assignment)
    context = render_identity["cb_serialized_payload"]
    identity = "exact-tensor-identity"
    assignment_path = tmp_path / "candidate_cb.json"
    assignment_path.write_text(json.dumps({
        "schema": "prismaquant.allocator.pareto_assignment.v1",
        "cb_serialized_payload": context,
        "cb_render_identity": render_identity,
        "cb_serialized_identities": {
            qname: identity,
        },
        "assignment": assignment,
    }))
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({
        "results": [{
            "label": "candidate_cb",
            "path": str(assignment_path),
            "bpp": 2.3,
            "last_token_kl": 0.01,
            "format_counts": {"NVFP4_CB_K16": 1},
        }],
    }))
    layer_config = tmp_path / "layer_config.json"
    assignment_out = tmp_path / "selected_assignment.json"
    summary = tmp_path / "selection.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.select_validated_frontier",
            "--validation-json",
            str(validation_path),
            "--mode",
            "best-kl",
            "--output-layer-config",
            str(layer_config),
            "--output-assignment",
            str(assignment_out),
            "--output-summary",
            str(summary),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    emitted_layer = json.loads(layer_config.read_text())
    assert emitted_layer["__prismaquant__"]["cb_serialized_payload"] == context
    assert emitted_layer["__prismaquant__"][
        "cb_render_identity"
    ] == render_identity
    assert emitted_layer[qname][
        "cb_serialized_identity"
    ] == identity
    emitted_assignment = json.loads(assignment_out.read_text())
    assert emitted_assignment["cb_serialized_payload"] == context
    assert emitted_assignment["cb_render_identity"] == render_identity
    assert emitted_assignment["cb_serialized_identities"] == {
        qname: identity,
    }


def test_selector_uses_validator_resolved_full_assignment_not_raw_delta(tmp_path):
    body = "model.layers.0.self_attn.q_proj"
    visual = "model.visual.blocks.0.mlp.fc1"
    assignment = {
        body: "NVFP4_CB_K16",
        visual: "NVFP4_CB_K16",
    }
    render_identity = _complete_cb_render_identity(assignment)
    context = render_identity["cb_serialized_payload"]
    raw_path = tmp_path / "raw_delta.json"
    raw_path.write_text(json.dumps({
        "assignment": {body: "NVFP4_CB_K16"},
        "cb_serialized_payload": context,
        "cb_serialized_identities": {body: "body-identity"},
    }))
    resolved = {
        "schema": "prismaquant.validated_resolved_assignment.v1",
        "source_path": str(raw_path),
        "assignment": assignment,
        "cb_serialized_payload": context,
        "cb_render_identity": render_identity,
        "cb_serialized_identities": {
            body: "body-identity",
            visual: "visual-identity",
        },
    }
    resolved["assignment_sha256"] = assignment_serialization_sha256(
        resolved["assignment"]
    )
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({
        "results": [{
            "label": "candidate",
            "path": str(raw_path),
            "bpp": 2.3,
            "last_token_kl": 0.01,
            "assignment_sha256": resolved["assignment_sha256"],
            "resolved_assignment_payload": resolved,
            "resolved_assignment_payload_sha256": (
                _canonical_payload_sha256(resolved)
            ),
        }],
    }))
    layer_config = tmp_path / "layer_config.json"
    assignment_out = tmp_path / "selected_assignment.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.select_validated_frontier",
            "--validation-json",
            str(validation_path),
            "--mode",
            "best-kl",
            "--output-layer-config",
            str(layer_config),
            "--output-assignment",
            str(assignment_out),
            "--output-summary",
            str(tmp_path / "selection.json"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    emitted = json.loads(assignment_out.read_text())
    assert emitted["assignment"] == resolved["assignment"]
    assert emitted["cb_serialized_identities"] == (
        resolved["cb_serialized_identities"]
    )
    assert emitted["cb_render_identity"] == render_identity
    emitted_layer = json.loads(layer_config.read_text())
    assert emitted_layer[visual]["cb_serialized_identity"] == "visual-identity"


@pytest.mark.parametrize("splice", ["assignment", "cb_render_identity"])
def test_selector_rejects_spliced_resolved_assignment_payload(tmp_path, splice):
    qname = "model.layers.0.self_attn.q_proj"
    assignment = {qname: "NVFP4_CB_K16"}
    render_identity = _complete_cb_render_identity(assignment)
    raw_path = tmp_path / "candidate.json"
    raw_path.write_text(json.dumps({"assignment": assignment}))
    resolved = {
        "schema": "prismaquant.validated_resolved_assignment.v1",
        "source_path": str(raw_path),
        "assignment": assignment,
        "assignment_sha256": assignment_serialization_sha256(assignment),
        "cb_serialized_payload": render_identity["cb_serialized_payload"],
        "cb_render_identity": render_identity,
        "cb_serialized_identities": {qname: "exact-tensor-identity"},
    }
    validator_payload_sha256 = _canonical_payload_sha256(resolved)
    spliced_payload = json.loads(json.dumps(resolved))
    if splice == "assignment":
        spliced_payload["assignment"][qname] = "NVFP4_CB_K14"
    else:
        spliced_payload["cb_render_identity"][
            "col_weights_content_sha256"
        ][qname] = "0" * 64

    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({
        "results": [{
            "label": "spliced",
            "path": str(raw_path),
            "bpp": 2.3,
            "last_token_kl": 0.01,
            "assignment_sha256": resolved["assignment_sha256"],
            "resolved_assignment_payload": spliced_payload,
            "resolved_assignment_payload_sha256": validator_payload_sha256,
        }],
    }))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.select_validated_frontier",
            "--validation-json",
            str(validation_path),
            "--mode",
            "best-kl",
            "--output-layer-config",
            str(tmp_path / "layer_config.json"),
            "--output-assignment",
            str(tmp_path / "selected_assignment.json"),
            "--output-summary",
            str(tmp_path / "selection.json"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "resolved measured assignment payload does not match the identity "
        "bound to its KL result"
    ) in (result.stderr + result.stdout)


def test_selector_carries_non_cb_whole_artifact_budget_to_layer_config(tmp_path):
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
    }
    budget = {
        "schema": "prismaquant.whole_artifact_budget.v2",
        "scope": "all_regular_files_recursive",
        "budget_bytes": 10_000,
        "selection_tensor_payload_bytes": 8_000,
        "selection_non_tensor_reserve_bytes": 1_000,
        "selection_whole_artifact_upper_bound_bytes": 9_000,
        "selection_assignment_sha256": assignment_serialization_sha256(
            assignment
        ),
        "selection_contract": (
            "tensor_payload_plus_operator_supplied_non_tensor_reserve"
        ),
        "final_contract": "stat_all_regular_files_recursive_fail_closed",
    }
    assignment_path = tmp_path / "candidate.json"
    assignment_path.write_text(json.dumps({
        "whole_artifact_budget": budget,
        "assignment": assignment,
    }))
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({
        "results": [{
            "label": "candidate",
            "path": str(assignment_path),
            "bpp": 4.5,
            "last_token_kl": 0.01,
        }],
    }))
    layer_config = tmp_path / "layer_config.json"
    assignment_out = tmp_path / "selected_assignment.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.select_validated_frontier",
            "--validation-json",
            str(validation_path),
            "--mode",
            "best-kl",
            "--output-layer-config",
            str(layer_config),
            "--output-assignment",
            str(assignment_out),
            "--output-summary",
            str(tmp_path / "selection.json"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    emitted_layer = json.loads(layer_config.read_text())
    assert emitted_layer["__prismaquant__"]["whole_artifact_budget"] == budget
    assert "cb_serialized_payload" not in emitted_layer["__prismaquant__"]
    emitted_assignment = json.loads(assignment_out.read_text())
    assert emitted_assignment["whole_artifact_budget"] == budget


def test_selector_rejects_global_cb_stamp_without_per_layer_identities(tmp_path):
    assignment_path = tmp_path / "candidate_cb.json"
    assignment_path.write_text(json.dumps({
        "cb_serialized_payload": _production_cb_stamp(),
        "assignment": {
            "model.layers.0.self_attn.q_proj": "NVFP4_CB_K16",
        },
    }))
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({
        "results": [{
            "label": "candidate_cb",
            "path": str(assignment_path),
            "bpp": 2.3,
            "last_token_kl": 0.01,
        }],
    }))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.select_validated_frontier",
            "--validation-json",
            str(validation_path),
            "--mode",
            "best-kl",
            "--output-layer-config",
            str(tmp_path / "layer_config.json"),
            "--output-assignment",
            str(tmp_path / "selected_assignment.json"),
            "--output-summary",
            str(tmp_path / "selection.json"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "global serialized-payload context but no per-layer identities" in (
        result.stderr + result.stdout
    )


def test_selector_rejects_cb_assignment_with_no_serialization_metadata(tmp_path):
    assignment_path = tmp_path / "candidate_cb.json"
    assignment_path.write_text(json.dumps({
        "assignment": {
            "model.layers.0.self_attn.q_proj": "NVFP4_CB_K16",
        },
    }))
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({
        "results": [{
            "label": "candidate_cb",
            "path": str(assignment_path),
            "bpp": 2.3,
            "last_token_kl": 0.01,
        }],
    }))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.select_validated_frontier",
            "--validation-json",
            str(validation_path),
            "--mode",
            "best-kl",
            "--output-layer-config",
            str(tmp_path / "layer_config.json"),
            "--output-assignment",
            str(tmp_path / "selected_assignment.json"),
            "--output-summary",
            str(tmp_path / "selection.json"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing its global serialized-payload context" in (
        result.stderr + result.stdout
    )


@pytest.mark.parametrize(
    "case,stamp_budget,stamp_upper,expected_error",
    [
        (
            "missing_stamp",
            None,
            None,
            "has no whole_artifact_budget stamp",
        ),
        (
            "wrong_budget",
            11_000,
            9_000,
            "budget differs from --target-disk-gb",
        ),
        (
            "wrong_upper",
            10_000,
            8_000,
            "upper bound does not reconcile",
        ),
    ],
)
def test_budget_selector_requires_reconciled_export_gate(
    tmp_path, case, stamp_budget, stamp_upper, expected_error,
):
    assignment_payload = {
        "assignment": {
            "model.layers.0.self_attn.q_proj": "NVFP4",
        },
    }
    if stamp_budget is not None:
        reserve = 1_000
        assignment_payload["whole_artifact_budget"] = {
            "schema": "prismaquant.whole_artifact_budget.v2",
            "scope": "all_regular_files_recursive",
            "budget_bytes": stamp_budget,
            "selection_tensor_payload_bytes": stamp_upper - reserve,
            "selection_non_tensor_reserve_bytes": reserve,
            "selection_whole_artifact_upper_bound_bytes": stamp_upper,
            "selection_assignment_sha256": assignment_serialization_sha256(
                assignment_payload["assignment"]
            ),
            "selection_contract": (
                "tensor_payload_plus_operator_supplied_non_tensor_reserve"
            ),
            "final_contract": (
                "stat_all_regular_files_recursive_fail_closed"
            ),
        }
    assignment_path = tmp_path / f"{case}.json"
    assignment_path.write_text(json.dumps(assignment_payload))
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({
        "results": [{
            "label": case,
            "path": str(assignment_path),
            "bpp": 4.5,
            "last_token_kl": 0.01,
            "whole_artifact_upper_bound_bytes": 9_000,
        }],
    }))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.select_validated_frontier",
            "--validation-json",
            str(validation_path),
            "--mode",
            "budget",
            "--target-disk-gb",
            "0.00001",  # floor(1e-5 * 1e9) = 10,000 bytes
            "--tail-veto",
            "none",
            "--output-layer-config",
            str(tmp_path / "layer_config.json"),
            "--output-assignment",
            str(tmp_path / "selected_assignment.json"),
            "--output-summary",
            str(tmp_path / "selection.json"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert expected_error in (result.stderr + result.stdout)


def test_selector_rejects_stale_cb_stamp_on_non_cb_assignment(tmp_path):
    assignment_path = tmp_path / "candidate.json"
    assignment_path.write_text(json.dumps({
        "cb_serialized_payload": _production_cb_stamp(),
        "assignment": {
            "model.layers.0.self_attn.q_proj": "NVFP4",
        },
    }))
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({
        "results": [{
            "label": "candidate",
            "path": str(assignment_path),
            "bpp": 4.5,
            "last_token_kl": 0.01,
        }],
    }))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.select_validated_frontier",
            "--validation-json",
            str(validation_path),
            "--mode",
            "best-kl",
            "--output-layer-config",
            str(tmp_path / "layer_config.json"),
            "--output-assignment",
            str(tmp_path / "selected_assignment.json"),
            "--output-summary",
            str(tmp_path / "selection.json"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "global serialized-payload context but no per-layer identities" in (
        result.stderr + result.stdout
    )


def test_select_validated_frontier_diagnostics_include_dominated_rows(tmp_path):
    assignment_paths = {}
    for label in ("a", "b", "c"):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({
            "assignment": {
                "model.layers.0.self_attn.q_proj": "BF16",
            },
        }))
        assignment_paths[label] = path

    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({
        "results": [
            {"label": "a", "path": str(assignment_paths["a"]), "bpp": 4.5,
             "last_token_kl": 0.10, "mse": {"predicted_dloss_sum": 2.0}},
            # Dominated by a on both bpp and KL, but surrogate ranks it best.
            {"label": "b", "path": str(assignment_paths["b"]), "bpp": 4.6,
             "last_token_kl": 0.30, "mse": {"predicted_dloss_sum": 1.0}},
            {"label": "c", "path": str(assignment_paths["c"]), "bpp": 5.0,
             "last_token_kl": 0.05, "mse": {"predicted_dloss_sum": 3.0}},
        ],
    }))
    layer_config = tmp_path / "layer_config.json"
    assignment_out = tmp_path / "selected_assignment.json"
    summary_path = tmp_path / "selection.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "prismaquant.select_validated_frontier",
            "--validation-json",
            str(validation_path),
            "--mode",
            "best-kl",
            "--output-layer-config",
            str(layer_config),
            "--output-assignment",
            str(assignment_out),
            "--output-summary",
            str(summary_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    summary = json.loads(summary_path.read_text())
    assert summary["n_results"] == 3
    assert summary["n_frontier"] == 2
    assert summary["surrogate_spearman"] is not None
    inversion = summary["surrogate_worst_rank_inversion"]
    assert inversion["predicted_best_label"] == "b"
    assert inversion["predicted_worse_label"] == "c"


def test_load_assignment_canonicalizes_autoround_dicts(tmp_path):
    # Regression: AutoRound-style dict entries used to be silently
    # stringified ("{'DATA_TYPE': 'NV_FP', ...}") instead of parsed.
    from prismaquant.select_validated_frontier import _load_assignment

    path = tmp_path / "assignment.json"
    path.write_text(json.dumps({
        "model.layers.0.mlp.experts.gate_up_proj": {
            "data_type": "nv_fp", "bits": 4, "group_size": 16, "sym": True,
        },
        "model.layers.0.self_attn.q_proj": {
            "data_type": "fp8_e4m3", "bits": 8, "group_size": 0,
        },
        "model.layers.0.self_attn.o_proj": "bf16",
        "model.layers.1.self_attn.o_proj": "FP8_SOURCE",
    }))
    assignment = _load_assignment(path)
    assert assignment == {
        "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
        "model.layers.0.self_attn.q_proj": "FP8_E4M3",
        "model.layers.0.self_attn.o_proj": "BF16",
        "model.layers.1.self_attn.o_proj": "FP8_SOURCE",
    }


def test_log_error_floors_non_positive_at_min_positive():
    # A measured KL <= 0 is "at the floor of measurement", not a million
    # times better than the best real point (audit §3.1).
    values = [0.10, 0.01, 0.0, -1e-9]
    logs = _log_error_values(values)
    assert logs[0] == math.log10(0.10)
    assert logs[1] == math.log10(0.01)
    # Non-positive values land exactly 0 decades below the smallest real point.
    assert logs[2] == math.log10(0.01)
    assert logs[3] == math.log10(0.01)


def test_kneedle_zero_kl_point_does_not_flip_knee_to_worst_point():
    # Audit §3.1 synthetic: a decreasing frontier {4.0/0.10 .. 6.0/0.010} plus
    # a near-passthrough point measuring KL 0.0. The old 1e-6 floor put that
    # point 6 fake decades below the curve, compressing the real points into a
    # flat band and flipping the knee to the lowest-bpp (worst) candidate.
    base = [
        {"label": "p40", "bpp": 4.0, "kl": 0.10},
        {"label": "p45", "bpp": 4.5, "kl": 0.055},
        {"label": "p50", "bpp": 5.0, "kl": 0.030},
        {"label": "p55", "bpp": 5.5, "kl": 0.017},
        {"label": "p60", "bpp": 6.0, "kl": 0.010},
    ]
    with_zero = base + [{"label": "p65", "bpp": 6.5, "kl": 0.0}]

    assert base[_kneedle_convex_decreasing(base)]["label"] == "p50"
    knee = with_zero[_kneedle_convex_decreasing(with_zero)]
    # The zero point reads as "at the measurement floor" (== 0.010), so the
    # curve is flat past 6.0 and the knee lands where it reaches the floor —
    # emphatically not at the curve start.
    assert knee["label"] != "p40"
    assert knee["label"] == "p60"


def _loo_rows():
    return [
        {"label": "a", "path": "a", "bpp": 4.0, "kl": 0.30},
        {"label": "b", "path": "b", "bpp": 4.5, "kl": 0.12},
        # Dominated by b; must re-enter the envelope when b is dropped.
        {"label": "b2", "path": "b2", "bpp": 4.6, "kl": 0.125},
        {"label": "c", "path": "c", "bpp": 5.0, "kl": 0.10},
        {"label": "d", "path": "d", "bpp": 5.5, "kl": 0.095},
        {"label": "e", "path": "e", "bpp": 6.0, "kl": 0.09},
    ]


def test_leave_one_out_rebuilds_envelope_from_all_rows():
    rows = _loo_rows()
    frontier = _frontier_from_rows(rows)
    assert [r["label"] for r in frontier] == ["a", "b", "c", "d", "e"]
    selected = frontier[_kneedle_convex_decreasing(frontier)]
    assert selected["label"] == "b"

    rebuilt = leave_one_out_kneedle_diagnostic(
        frontier, selected, all_rows=rows,
    )
    frozen = leave_one_out_kneedle_diagnostic(frontier, selected)
    rebuilt_picks = {p["dropped_label"]: p["selected_label"] for p in rebuilt["picks"]}
    frozen_picks = {p["dropped_label"]: p["selected_label"] for p in frozen["picks"]}
    # Dropping the knee lets the dominated interior point b2 re-enter the
    # envelope and win the kneedle; the frozen envelope could never see it.
    assert rebuilt_picks["b"] == "b2"
    assert frozen_picks["b"] == "c"


def test_leave_one_out_stability_tolerance_from_repeat_stderr():
    rows = [
        {"label": "a", "path": "a", "bpp": 4.0, "kl": 0.400, "kl_stderr": 2e-3},
        {"label": "b", "path": "b", "bpp": 4.5, "kl": 0.200, "kl_stderr": 2e-3},
        {"label": "c", "path": "c", "bpp": 5.0, "kl": 0.1000, "kl_stderr": 2e-3},
        {"label": "d", "path": "d", "bpp": 5.2, "kl": 0.0990, "kl_stderr": 2e-3},
        {"label": "e", "path": "e", "bpp": 6.0, "kl": 0.0950, "kl_stderr": 2e-3},
        {"label": "f", "path": "f", "bpp": 6.5, "kl": 0.0930, "kl_stderr": 2e-3},
    ]
    frontier = _frontier_from_rows(rows)
    selected = frontier[_kneedle_convex_decreasing(frontier)]
    assert selected["label"] == "c"

    diag = leave_one_out_kneedle_diagnostic(
        frontier, selected, tolerance_bpp=0.6, all_rows=rows,
    )
    # LOO shift (c -> d, |dKL| = 0.001) is within the knee's measured repeat
    # stderr (0.002): indistinguishable from measurement noise -> stable.
    assert diag["stability_tolerance_source"] == "repeat_stderr"
    assert diag["kl_stability_tolerance"] == 2e-3
    assert diag["max_kl_shift"] <= 2e-3
    assert diag["stable"] is True

    # Without repeat data there is no measured noise scale: strict 0.
    strict_rows = [
        {k: v for k, v in row.items() if k != "kl_stderr"} for row in rows
    ]
    strict_frontier = _frontier_from_rows(strict_rows)
    strict_selected = strict_frontier[_kneedle_convex_decreasing(strict_frontier)]
    strict = leave_one_out_kneedle_diagnostic(
        strict_frontier, strict_selected, tolerance_bpp=0.6, all_rows=strict_rows,
    )
    assert strict["stability_tolerance_source"] == "strict"
    assert strict["kl_stability_tolerance"] == 0.0
    assert strict["stable"] is False

    # An explicit noise floor always wins over the stderr.
    floored = leave_one_out_kneedle_diagnostic(
        frontier, selected, tolerance_bpp=0.6, kl_noise_floor=0.05, all_rows=rows,
    )
    assert floored["stability_tolerance_source"] == "kl_noise_floor"
    assert floored["kl_stability_tolerance"] == 0.05


def test_load_assignment_unwraps_assignment_key(tmp_path):
    from prismaquant.select_validated_frontier import _load_assignment

    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps({
        "schema": "prismaquant.validated_frontier_assignment.v1",
        "assignment": {"model.layers.0.mlp.up_proj": "nvfp4"},
    }))
    assert _load_assignment(path) == {"model.layers.0.mlp.up_proj": "NVFP4"}


# --------------------------------------------------------------------------
# R9 / D1 tail veto
# --------------------------------------------------------------------------

def _tail_rows():
    """A frontier where mean KL falls monotonically but the tail does not.

    ``c`` is the D1 shape: the best mean KL on the curve, bought with a p99
    that is 3x the incumbent's. Selecting on the mean alone ships it.
    """
    return [
        {"label": "a", "path": "a", "bpp": 4.5, "kl": 0.30,
         "kl_p99": 0.90, "kl_max": 1.10, "nll_p99": 3.0},
        {"label": "b", "path": "b", "bpp": 5.0, "kl": 0.20,
         "kl_p99": 0.60, "kl_max": 0.70, "nll_p99": 2.8},
        {"label": "c", "path": "c", "bpp": 5.5, "kl": 0.10,
         "kl_p99": 1.80, "kl_max": 2.40, "nll_p99": 4.5},
        {"label": "d", "path": "d", "bpp": 6.0, "kl": 0.05,
         "kl_p99": 0.40, "kl_max": 0.50, "nll_p99": 2.5},
    ]


def test_tail_veto_none_mode_is_frontier_identity():
    """`--tail-veto none` must reproduce the pre-R9 envelope exactly."""
    rows = _tail_rows()
    baseline = _frontier_from_rows(rows)
    assert [r["label"] for r in baseline] == ["a", "b", "c", "d"]
    for explicit_off in (None, "none"):
        vetoed = []
        again = _frontier_from_rows(rows, tail_veto=explicit_off, vetoed=vetoed)
        assert again == baseline
        assert vetoed == []


def test_tail_veto_refuses_a_mean_win_that_regresses_the_tail():
    rows = _tail_rows()
    vetoed = []
    frontier = _frontier_from_rows(rows, tail_veto="kl_p99", vetoed=vetoed)
    # c wins on the mean and loses on p99 -> refused; d still enters, because
    # its tail improves on the last ADMITTED point (b), not on the vetoed c.
    assert [r["label"] for r in frontier] == ["a", "b", "d"]
    assert [r["label"] for r in vetoed] == ["c"]
    assert vetoed[0]["veto_reason"] == "tail_regression"
    assert vetoed[0]["veto_column"] == "kl_p99"
    assert vetoed[0]["veto_value"] == 1.80
    assert vetoed[0]["veto_incumbent"] == 0.60


def test_tail_veto_eta_admits_within_slack():
    rows = [
        {"label": "a", "path": "a", "bpp": 4.5, "kl": 0.30, "kl_p99": 1.00},
        {"label": "b", "path": "b", "bpp": 5.0, "kl": 0.20, "kl_p99": 1.05},
    ]
    assert [r["label"] for r in _frontier_from_rows(rows, tail_veto="kl_p99")] == ["a"]
    admitted = _frontier_from_rows(rows, tail_veto="kl_p99", tail_eta=0.10)
    assert [r["label"] for r in admitted] == ["a", "b"]


def test_tail_veto_reports_missing_column_rather_than_admitting():
    rows = [
        {"label": "a", "path": "a", "bpp": 4.5, "kl": 0.30, "nll_p99": 3.0},
        {"label": "b", "path": "b", "bpp": 5.0, "kl": 0.20},
    ]
    vetoed = []
    frontier = _frontier_from_rows(rows, tail_veto="nll_p99", vetoed=vetoed)
    assert [r["label"] for r in frontier] == ["a"]
    assert vetoed[0]["veto_reason"] == "tail_missing"


def test_measured_rows_pass_through_tail_columns():
    results = [{
        "label": "a", "path": "a.json", "bpp": 5.0, "kl_mean": 0.10,
        "kl_p95": 0.3, "kl_p99": 0.5, "kl_max": 0.9,
        "nll_mean": 2.0, "nll_p99": 3.0,
    }]
    row = measured_rows(results)[0]
    assert row["kl"] == 0.10  # kl_mean resolves as the mean metric
    assert row["kl_p95"] == 0.3
    assert row["kl_p99"] == 0.5
    assert row["kl_max"] == 0.9
    assert row["nll_mean"] == 2.0
    assert row["nll_p99"] == 3.0


def test_select_frontier_point_threads_the_veto():
    results = [
        {"label": r["label"], "path": f"{r['label']}.json", "bpp": r["bpp"],
         "last_token_kl": r["kl"], "kl_p99": r["kl_p99"]}
        for r in _tail_rows()
    ]
    vetoed = []
    selected, frontier = select_frontier_point(
        results, mode="best-kl", tail_veto="kl_p99", vetoed=vetoed)
    assert selected["label"] == "d"
    assert [r["label"] for r in frontier] == ["a", "b", "d"]
    assert [r["label"] for r in vetoed] == ["c"]

    # Off, the mean-only frontier keeps c and best-kl still lands on d.
    _sel_off, frontier_off = select_frontier_point(results, mode="best-kl")
    assert [r["label"] for r in frontier_off] == ["a", "b", "c", "d"]


def _write_validation(tmp_path, rows, columns=("kl_p99",)):
    results = []
    for row in rows:
        path = tmp_path / f"{row['label']}.json"
        path.write_text(json.dumps({
            "schema": "prismaquant.allocator.pareto_assignment.v1",
            "assignment": {"model.layers.0.mlp.down_proj": "NVFP4"},
        }))
        results.append({
            "label": row["label"], "path": str(path), "bpp": row["bpp"],
            "last_token_kl": row["kl"],
            **{c: row[c] for c in columns if c in row},
        })
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({"results": results}))
    return validation_path


def _run_select(tmp_path, validation_path, extra, tag="run"):
    summary = tmp_path / f"selection-{tag}.json"
    subprocess.run(
        [sys.executable, "-m", "prismaquant.select_validated_frontier",
         "--validation-json", str(validation_path),
         "--mode", "best-kl",
         "--output-layer-config", str(tmp_path / "lc.json"),
         "--output-assignment", str(tmp_path / "sa.json"),
         "--output-summary", str(summary), *extra],
        cwd=Path(__file__).resolve().parents[1], check=True,
    )
    return json.loads(summary.read_text())


def test_tail_veto_cli_default_is_on_with_kl_max(tmp_path):
    """Ruled 2026-07-30: the veto ships DEFAULT-ON with kl_max as the contract."""
    assert DEFAULT_TAIL_VETO == "kl_max"
    assert DEFAULT_TAIL_ETA == "auto"
    validation_path = _write_validation(
        tmp_path, _tail_rows(), columns=("kl_p99", "kl_max"))
    default = _run_select(tmp_path, validation_path, [], tag="default")
    assert default["tail_veto"]["column"] == "kl_max"
    assert default["tail_veto"]["eta"] == "auto"
    assert default["tail_veto"]["eta_mode"] == "auto"
    assert default["tail_veto"]["inert_reason"] is None
    # c wins the mean and blows the worst-sequence tail (2.40 vs 0.70): refused.
    assert [r["label"] for r in default["vetoed_rows"]] == ["c"]
    assert [r["label"] for r in default["frontier"]] == ["a", "b", "d"]
    assert default["vetoed_rows"][0]["veto_column"] == "kl_max"
    # No per-repeat tails on these rows -> the derived slack degrades to strict 0.
    assert {e["source"] for e in default["tail_veto"]["eta_resolved"]} == {"absent"}


def test_tail_veto_cli_none_mode_restores_the_pre_r9_envelope(tmp_path):
    validation_path = _write_validation(
        tmp_path, _tail_rows(), columns=("kl_p99", "kl_max"))
    off = _run_select(tmp_path, validation_path, ["--tail-veto", "none"],
                      tag="off")
    assert off["tail_veto"]["column"] is None
    assert off["vetoed_rows"] == []
    assert [r["label"] for r in off["frontier"]] == ["a", "b", "c", "d"]


def test_tail_veto_cli_goes_inert_on_a_pre_r9_validation_json(tmp_path):
    """Default-on must not turn a tail-less (pre-R9) input into an empty frontier."""
    validation_path = _write_validation(tmp_path, _tail_rows(), columns=())
    out = _run_select(tmp_path, validation_path, [], tag="inert")
    assert out["tail_veto"]["column"] == "kl_max"
    assert out["tail_veto"]["inert_reason"] == "tail_column_absent_on_every_row"
    assert out["vetoed_rows"] == []
    assert [r["label"] for r in out["frontier"]] == ["a", "b", "c", "d"]


def test_tail_veto_cli_records_vetoed_rows(tmp_path):
    validation_path = _write_validation(tmp_path, _tail_rows())
    on = _run_select(tmp_path, validation_path,
                     ["--tail-veto", "kl_p99", "--tail-eta", "0.0"], tag="p99")
    assert on["tail_veto"]["column"] == "kl_p99"
    assert on["tail_veto"]["eta"] == 0.0
    assert on["tail_veto"]["eta_mode"] == "explicit"
    assert on["tail_veto"]["n_vetoed"] == 1
    assert [r["label"] for r in on["vetoed_rows"]] == ["c"]
    assert [r["label"] for r in on["frontier"]] == ["a", "b", "d"]


def test_tail_veto_cli_explicit_eta_beats_auto(tmp_path):
    rows = [
        {"label": "a", "path": "a", "bpp": 4.5, "kl": 0.30, "kl_max": 1.00},
        {"label": "b", "path": "b", "bpp": 5.0, "kl": 0.20, "kl_max": 1.05},
    ]
    validation_path = _write_validation(tmp_path, rows, columns=("kl_max",))
    strict = _run_select(tmp_path, validation_path, [], tag="strict")
    assert [r["label"] for r in strict["frontier"]] == ["a"]
    loose = _run_select(tmp_path, validation_path, ["--tail-eta", "0.10"],
                        tag="loose")
    assert loose["tail_veto"]["eta"] == 0.10
    assert [r["label"] for r in loose["frontier"]] == ["a", "b"]


# ---- the derived slack (--tail-eta auto) --------------------------------


def test_tail_eta_auto_is_the_between_repeat_relative_stderr():
    repeats = [1.0, 1.1, 0.9, 1.0]
    eta, source = tail_eta_auto({"kl_max_repeats": repeats}, "kl_max")
    assert source == "derived"
    mean = sum(repeats) / len(repeats)
    var = sum((v - mean) ** 2 for v in repeats) / (len(repeats) - 1)
    assert eta == math.sqrt(var) / math.sqrt(len(repeats)) / mean
    assert eta > 0.0


def test_tail_eta_auto_degrades_to_strict_zero_without_a_spread():
    assert tail_eta_auto({"kl_max_repeats": [1.0]}, "kl_max") == (0.0, "single_repeat")
    assert tail_eta_auto({}, "kl_max") == (0.0, "absent")
    assert tail_eta_auto({"kl_max_repeats": [0.0, 0.0]}, "kl_max") == (0.0, "degenerate")


def test_tail_eta_auto_admits_inside_the_measured_noise_and_refuses_outside():
    """The slack is the tail's own noise: 4% moves are noise, 100% is a regression."""
    rows = [
        {"label": "a", "path": "a", "bpp": 4.5, "kl": 0.30, "kl_max": 1.00,
         "kl_max_repeats": [1.0, 1.1, 0.9, 1.0]},
        {"label": "b", "path": "b", "bpp": 5.0, "kl": 0.20, "kl_max": 1.02},
        {"label": "c", "path": "c", "bpp": 5.5, "kl": 0.10, "kl_max": 2.00},
    ]
    vetoed = []
    frontier = _frontier_from_rows(
        rows, tail_veto="kl_max", tail_eta="auto", vetoed=vetoed)
    assert [r["label"] for r in frontier] == ["a", "b"]
    assert [r["label"] for r in vetoed] == ["c"]
    assert vetoed[0]["veto_eta_source"] == "absent"  # b carries no repeats
    # With eta hard-zero, b's 2% is refused too: the derived slack is doing work.
    strict = _frontier_from_rows(rows, tail_veto="kl_max", tail_eta=0.0)
    assert [r["label"] for r in strict] == ["a"]


def test_tail_veto_inert_reason_only_fires_when_no_row_carries_the_column():
    rows = _tail_rows()
    assert tail_veto_inert_reason(rows, "kl_max") is None
    assert tail_veto_inert_reason(rows, None) is None
    bare = [{"label": "a", "bpp": 4.5, "kl": 0.3}]
    assert tail_veto_inert_reason(bare, "kl_max") == "tail_column_absent_on_every_row"
    # Inert -> the envelope is the mean-only one, not an empty frontier.
    assert len(_frontier_from_rows(bare, tail_veto="kl_max")) == 1


# ---------------------------------------------------------------------------
# Rate-axis gate (#117): a Tessera pick is a candidate until a byte-matched
# uniform arm corroborates it
# ---------------------------------------------------------------------------

#: A real attested rung, so the fixture allocates over the axis, not a mock.
_TESSERA_RUNG = "TESSERA_E2M1_K2_R896"
#: The 2026-09-02 receipt's own candidate bytes (1761722368 bits), so the
#: refusal names a measurement rather than an invented number.
_RECEIPT_CANDIDATE_BYTES = 1761722368 // 8


def _rate_axis_cli_fixture(tmp_path, *, formats):
    assignment_path = tmp_path / "rate_axis_candidate.json"
    assignment_path.write_text(json.dumps({
        "schema": "prismaquant.allocator.pareto_assignment.v1",
        "assignment": {
            "model.layers.0.self_attn.q_proj": formats[0],
            "model.layers.0.mlp.down_proj": formats[1],
        },
    }))
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({
        "results": [
            {
                "label": "alloc_4.0",
                "path": str(assignment_path),
                "bpp": 4.0,
                "last_token_kl": 0.02,
                "whole_artifact_upper_bound_bytes": _RECEIPT_CANDIDATE_BYTES,
                "format_counts": {formats[0]: 1, formats[1]: 1},
            },
            {
                "label": "alloc_5.0",
                "path": str(assignment_path),
                "bpp": 5.0,
                "last_token_kl": 0.015,
                "whole_artifact_upper_bound_bytes": _RECEIPT_CANDIDATE_BYTES,
                "format_counts": {formats[0]: 1, formats[1]: 1},
            },
        ],
    }))
    layer_config = tmp_path / "layer_config.json"
    layer_config.write_text(json.dumps({
        "__prismaquant__": {"target_profile": "research"},
    }))
    return (
        validation_path,
        layer_config,
        tmp_path / "selected_assignment.json",
        tmp_path / "selection.json",
    )


def _run_selector(*args):
    return subprocess.run(
        [sys.executable, "-m", "prismaquant.select_validated_frontier", *args],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


def test_a_tessera_pick_refuses_without_a_byte_matched_uniform_arm(tmp_path):
    """The shipping point must not certify what it cannot corroborate.

    Measured 2026-09-02: a Tessera allocation served 2.00x worse KL than a
    byte-matched uniform arm while every check this stage owns passed, and an
    oracle over the same menu reaches only 0.941x of uniform -- so no ranking
    this stage can do closes the gap. The validated frontier re-ranks the
    allocator's own Pareto points and cannot see uniform beating the pick, so
    the pick ships as a candidate with the refusal naming the comparison, the
    bytes, and what would pass.
    """
    from prismaquant.select_validated_frontier import (
        RATE_AXIS_UNCERTIFIED_EXIT,
    )

    validation_path, layer_config, assignment_out, summary = (
        _rate_axis_cli_fixture(
            tmp_path, formats=(_TESSERA_RUNG, "NVFP4"))
    )
    proc = _run_selector(
        "--validation-json", str(validation_path),
        "--mode", "best-kl",
        "--output-layer-config", str(layer_config),
        "--output-assignment", str(assignment_out),
        "--output-summary", str(summary),
    )
    assert proc.returncode == RATE_AXIS_UNCERTIFIED_EXIT, proc.stdout + proc.stderr
    refused = proc.stdout + proc.stderr
    assert "REFUSED" in refused
    assert "byte-matched uniform" in refused
    assert "alloc_5.0" in refused
    assert str(_RECEIPT_CANDIDATE_BYTES) in refused

    # The recipe the control loop needs is still written -- the control is
    # built FROM the candidate plan -- but stamped as a candidate whose
    # uniform corroboration is outstanding, not as a selection.
    stamped = json.loads(summary.read_text())["uniform_control"]
    assert stamped["status"] == "outstanding"
    assert stamped["selected_label"] == "alloc_5.0"
    assert stamped["selected_artifact_bytes"] == _RECEIPT_CANDIDATE_BYTES
    assert stamped["rate_axis_formats"] == [_TESSERA_RUNG]
    meta = json.loads(layer_config.read_text())["__prismaquant__"]
    assert meta["uniform_control"]["status"] == "outstanding"
    # bpp provenance is factual and stays: this IS the validated-frontier
    # pick; what is refused is certifying it shippable.
    assert meta["selected_by"] == "validated_frontier:best-kl"


def test_a_format_menu_pick_still_certifies_without_a_uniform_arm(tmp_path):
    """Scope guard: no rung axis, no uniform rung, no refusal."""
    validation_path, layer_config, assignment_out, summary = (
        _rate_axis_cli_fixture(tmp_path, formats=("NVFP4", "BF16"))
    )
    proc = _run_selector(
        "--validation-json", str(validation_path),
        "--mode", "best-kl",
        "--output-layer-config", str(layer_config),
        "--output-assignment", str(assignment_out),
        "--output-summary", str(summary),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(summary.read_text())["uniform_control"] is None


def test_an_acknowledged_rate_axis_pick_builds_the_candidate(tmp_path):
    """The refusal is a gate, not a wall: the uniform control is built FROM
    the candidate plan, so a per-run acknowledgement lets the pipeline walk on
    to export while stamping the control outstanding -- acknowledged, never
    served -- in both output files. The wall sits at publication (#121)."""
    validation_path, layer_config, assignment_out, summary = (
        _rate_axis_cli_fixture(tmp_path, formats=(_TESSERA_RUNG, "NVFP4"))
    )
    proc = _run_selector(
        "--validation-json", str(validation_path),
        "--mode", "best-kl",
        "--acknowledge-outstanding-uniform-control",
        "run-2026-09-04-build-candidate",
        "--output-layer-config", str(layer_config),
        "--output-assignment", str(assignment_out),
        "--output-summary", str(summary),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ACKNOWLEDGED" in (proc.stdout + proc.stderr)
    stamped = json.loads(summary.read_text())["uniform_control"]
    assert stamped["status"] == "acknowledged"
    assert stamped["acknowledged_run"] == "run-2026-09-04-build-candidate"
    assert stamped["acknowledged_via"] == "flag"
    assert stamped["selected_label"] == "alloc_5.0"
    assert stamped["selected_artifact_bytes"] == _RECEIPT_CANDIDATE_BYTES
    meta = json.loads(layer_config.read_text())["__prismaquant__"]
    assert meta["uniform_control"]["status"] == "acknowledged"
    assert (meta["uniform_control"]["acknowledged_run"]
            == "run-2026-09-04-build-candidate")


def test_the_env_spelling_acknowledges_and_the_flag_wins(tmp_path, monkeypatch):
    """`PRISMAQUANT_ACKNOWLEDGE_OUTSTANDING_UNIFORM_CONTROL` carries the same
    run id for drivers that cannot pass the flag; an explicit flag wins."""
    monkeypatch.setenv(
        "PRISMAQUANT_ACKNOWLEDGE_OUTSTANDING_UNIFORM_CONTROL", "run-env-1")
    validation_path, layer_config, assignment_out, summary = (
        _rate_axis_cli_fixture(tmp_path, formats=(_TESSERA_RUNG, "NVFP4"))
    )
    both = _run_selector(
        "--validation-json", str(validation_path),
        "--mode", "best-kl",
        "--acknowledge-outstanding-uniform-control", "run-flag-1",
        "--output-layer-config", str(layer_config),
        "--output-assignment", str(assignment_out),
        "--output-summary", str(summary),
    )
    assert both.returncode == 0, both.stdout + both.stderr
    stamped = json.loads(summary.read_text())["uniform_control"]
    assert stamped["acknowledged_run"] == "run-flag-1"
    assert stamped["acknowledged_via"] == "flag"

    env_only = _run_selector(
        "--validation-json", str(validation_path),
        "--mode", "best-kl",
        "--output-layer-config", str(layer_config),
        "--output-assignment", str(assignment_out),
        "--output-summary", str(summary),
    )
    assert env_only.returncode == 0, env_only.stdout + env_only.stderr
    stamped = json.loads(summary.read_text())["uniform_control"]
    assert stamped["acknowledged_run"] == "run-env-1"
    assert stamped["acknowledged_via"] == "env"


def test_a_uniform_rate_axis_plan_is_its_own_control(tmp_path):
    """One format over every assigned unit IS a byte-matched uniform arm --
    refusing it and telling it to go build a uniform control would compare the
    plan against itself. It exits 0 with the status recording that there is
    nothing to corroborate."""
    validation_path, layer_config, assignment_out, summary = (
        _rate_axis_cli_fixture(
            tmp_path, formats=(_TESSERA_RUNG, _TESSERA_RUNG))
    )
    proc = _run_selector(
        "--validation-json", str(validation_path),
        "--mode", "best-kl",
        "--output-layer-config", str(layer_config),
        "--output-assignment", str(assignment_out),
        "--output-summary", str(summary),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    stamped = json.loads(summary.read_text())["uniform_control"]
    assert stamped["status"] == "not_applicable"
    assert stamped["reason"] == "uniform_assignment"
    assert stamped["rate_axis_formats"] == [_TESSERA_RUNG]
    meta = json.loads(layer_config.read_text())["__prismaquant__"]
    assert meta["uniform_control"]["status"] == "not_applicable"
