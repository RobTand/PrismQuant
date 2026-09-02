"""Wave-3 re-vet mechanisms that do not have a natural home elsewhere.

* R1  — the `budget` frontier pick: byte budget = constraint, measured KL =
        objective; kneedle stays the default without a card.
* R2  — cost-table provenance: every producer stamps the COST_MODE that made
        it, so `cost.pkl` cannot be silently reused across modes.
* R11 — the allocator's resolved serving profile travels in layer_config.json.
* R24 — the export prefetch `require` mode, and require_cuda_hot_path on the
        stages the CB/GGUF ladder invokes directly.
"""

import json
import re
from pathlib import Path

import pytest

from prismaquant import select_validated_frontier as svf
from prismaquant.layer_config import (
    LAYER_CONFIG_META_KEY,
    canonicalize_assignment,
    layer_config_metadata,
)
from prismaquant.schemas import validate_layer_config_payload

ROOT = Path(__file__).resolve().parents[1]
GB = 1_000_000_000.0


def _rows(tmp_path, spec):
    """spec: [(label, bpp, kl, gb)] -> results rows + priced assignment JSONs."""
    out = []
    for label, bpp, kl, gb in spec:
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({
            "label": label,
            "whole_artifact_upper_bound_bytes": int(gb * GB),
            "artifact_bytes": int(gb * GB),
            "artifact_byte_scope": (
                "selection_upper_bound_tensor_payload_plus_"
                "operator_non_tensor_reserve"
            ),
            "assignment": {"model.layers.0.self_attn.q_proj": "NVFP4"},
        }))
        out.append({"label": label, "path": str(path), "bpp": bpp, "kl": kl})
    return out


# --------------------------------------------------------------------- R1


def test_budget_pick_takes_min_kl_among_fitting_rows(tmp_path):
    rows = _rows(tmp_path, [
        ("a", 4.5, 0.040, 18.0),
        ("b", 5.0, 0.020, 20.0),
        ("c", 5.5, 0.010, 24.0),   # best KL, does NOT fit
    ])
    selected, frontier = svf.select_frontier_point(
        rows, mode="budget", budget_bytes=21.0 * GB)
    assert selected["label"] == "b"
    assert len(frontier) == 3


def test_budget_pick_requires_a_budget(tmp_path):
    rows = _rows(tmp_path, [("a", 4.5, 0.04, 18.0), ("b", 5.0, 0.02, 20.0)])
    with pytest.raises(ValueError, match="requires a byte budget"):
        svf.select_frontier_point(rows, mode="budget")


def test_budget_pick_refuses_unpriced_rows(tmp_path):
    rows = _rows(tmp_path, [("a", 4.5, 0.04, 18.0), ("b", 5.0, 0.02, 20.0)])
    (tmp_path / "b.json").write_text(json.dumps({"label": "b", "assignment": {}}))
    with pytest.raises(ValueError, match="unpriced"):
        svf.select_frontier_point(rows, mode="budget", budget_bytes=21.0 * GB)


def test_budget_pick_refuses_when_nothing_fits(tmp_path):
    rows = _rows(tmp_path, [("a", 4.5, 0.04, 18.0), ("b", 5.0, 0.02, 20.0)])
    with pytest.raises(ValueError, match="cheapest measured artifact"):
        svf.select_frontier_point(rows, mode="budget", budget_bytes=1.0 * GB)


def test_bytes_are_read_from_the_allocator_payload(tmp_path):
    rows = _rows(tmp_path, [("a", 4.5, 0.04, 18.0)])
    measured = svf.measured_rows(rows)
    assert measured[0]["artifact_bytes"] == int(18.0 * GB)


def test_kneedle_is_untouched_by_the_bytes_column(tmp_path):
    """R1 must be byte-identical when no card is given."""
    rows = _rows(tmp_path, [
        ("a", 4.5, 0.100, 18.0), ("b", 5.0, 0.030, 20.0),
        ("c", 5.5, 0.025, 24.0), ("d", 6.0, 0.024, 28.0),
    ])
    unpriced = [{k: v for k, v in row.items()} for row in rows]
    for row in unpriced:
        Path(row["path"]).write_text(json.dumps({"assignment": {}}))
    a, _ = svf.select_frontier_point(rows, mode="kneedle")
    b, _ = svf.select_frontier_point(unpriced, mode="kneedle")
    assert a["label"] == b["label"]


def test_run_pipeline_wires_the_card_into_both_stages():
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    assert '--target-disk-gb "$TARGET_DISK_GB"' in script
    # allocator AND selector both receive it
    assert script.count('--target-disk-gb "$TARGET_DISK_GB"') >= 2
    assert 'ALLOCATOR_BUDGET_ARGS' in script


def test_allocator_narrows_the_pareto_set_under_a_budget():
    src = (ROOT / "prismaquant" / "allocator.py").read_text()
    assert "byte-budget Pareto narrowing" in src
    # a computed narrowing, not a hardcoded rung count
    assert "keep_positions" in src
    assert re.search(
        r"whole_artifact_upper_bound_bytes.*<=.*budget_bytes_pareto", src
    )


# --------------------------------------------------------------------- R2


@pytest.mark.parametrize("module", [
    "incremental_measure_quant_cost",
    "production_render_cost",
    "aura_cost",
    "expert_empirical_cost",
])
def test_every_cost_producer_stamps_cost_mode(module):
    src = (ROOT / "prismaquant" / f"{module}.py").read_text()
    assert "--cost-mode" in src, f"{module} has no --cost-mode flag"
    assert "cost_mode" in src


def test_pipeline_passes_cost_mode_to_every_producer():
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    # Was >= 5 until 2026-09-02: one of the call sites was inside the
    # `EXPORT_CONTAINER=nvfp4_cb` arm and went with the Gridbook lane
    # (archive/gridbook_lane_2026-09-02/). The property -- every producer the
    # shell invokes is handed the cost mode -- is unchanged; only the number of
    # producers is smaller.
    assert script.count('--cost-mode "$COST_MODE"') >= 4
    assert "cost_table_reusable" in script
    # reuse of the ALLOCATOR cost table is conditional on the stamp
    assert 'if ! cost_table_reusable "$COST_PATH"' in script


def test_unstamped_cost_tables_are_reused_not_invalidated():
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    block = script.split("cost_table_reusable() {", 1)[1].split("\n}", 1)[0]
    assert "predates the R2 stamp" in block
    assert "return 0" in block


# -------------------------------------------------------------------- R11


def test_layer_config_metadata_round_trips_and_is_not_an_assignment_entry():
    payload = {
        "model.layers.0.self_attn.q_proj": {"data_type": "nv_fp", "bits": 4},
        LAYER_CONFIG_META_KEY: {"target_profile": "nvfp4_cb"},
    }
    validate_layer_config_payload(payload, "test")
    assert canonicalize_assignment(payload) == {
        "model.layers.0.self_attn.q_proj": "NVFP4"}
    assert layer_config_metadata(payload)["target_profile"] == "nvfp4_cb"


def test_reserved_metadata_must_be_an_object():
    with pytest.raises(Exception):
        validate_layer_config_payload({LAYER_CONFIG_META_KEY: "nvfp4_cb"}, "test")


def test_export_prefers_env_then_the_layer_config_stamp():
    src = (ROOT / "prismaquant" / "export_native_compressed.py").read_text()
    block = src.split("def _allocator_target_profile_for_audit", 1)[1].split("\n\n\n", 1)[0]
    env_at = block.index("PRISMAQUANT_TARGET_PROFILE")
    stamp_at = block.index("_ALLOCATOR_TARGET_PROFILE")
    assert env_at < stamp_at, "the env override must win over the stamp"


def test_frontier_selection_carries_the_stamp_forward():
    src = (ROOT / "prismaquant" / "select_validated_frontier.py").read_text()
    assert "read_layer_config_metadata" in src
    assert "LAYER_CONFIG_META_KEY" in src


# -------------------------------------------------------------------- R24


def test_export_prefetch_require_is_wired_on_the_native_lane():
    src = (ROOT / "prismaquant" / "export_native_compressed.py").read_text()
    assert "--production-cache-prefetch" in src
    assert "_PRODUCTION_CACHE_PREFETCH_MODE" in src
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    assert 'EXPORT_PRODUCTION_CACHE_PREFETCH:=require' in script
    assert '--production-cache-prefetch "$EXPORT_PRODUCTION_CACHE_PREFETCH"' in script


@pytest.mark.parametrize("module", [
    "incremental_probe",
    "incremental_measure_quant_cost",
    "aura_cost",
    "production_render_cost",
    # `export_nvfp4_cb` and `export_nvfp4_cb_streaming` left this list on
    # 2026-09-02 with the Gridbook lane (archive/gridbook_lane_2026-09-02/).
    # They were two of the entry points the ladder invoked directly, which is
    # precisely why they were pinned here; both modules are gone.
    "export_gguf",
    # the pre-existing callers, pinned so a refactor cannot drop them
    "build_production_cache",
    "expert_empirical_cost",
    "production_recache",
    "validate_assignments_kl",
])
def test_gpu_or_bust_guard_on_every_production_entrypoint(module):
    src = (ROOT / "prismaquant" / f"{module}.py").read_text()
    assert "require_cuda_hot_path" in src, (
        f"{module}.main() can run on CPU; the GGUF ladder invokes these "
        "directly, bypassing run-pipeline.sh's preflight")


def test_validated_frontier_selector_remains_cpu_safe():
    """Selection reads measured JSON and writes JSON; it launches no tensor
    work and must remain usable on orchestration/CI hosts without a GPU."""
    src = (ROOT / "prismaquant" / "select_validated_frontier.py").read_text()
    assert "require_cuda_hot_path" not in src
    assert "torch." not in src


# ---------------------------------------------------------------------- R6


def test_lane_preflight_is_wired_into_the_orchestrator():
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    assert "require_lane_supported" in script
    assert "TARGET_PROFILE_RESOLVED" in script
    preflight = script.split("if ! TARGET_PROFILE_RESOLVED=", 1)[1].split("\nfi\n", 1)[0]
    assert "raise SystemExit(2)" in preflight
