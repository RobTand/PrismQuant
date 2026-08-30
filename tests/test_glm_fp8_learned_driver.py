from __future__ import annotations

import importlib.util
from pathlib import Path


_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/fp8_learned_glm.py"
)
_SPEC = importlib.util.spec_from_file_location("fp8_learned_glm", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_DRIVER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DRIVER)


def _cell(population, fixed, learned, fixed_bpw=4.0, learned_bpw=4.1):
    arms = {}
    for rung in _DRIVER.RUNGS:
        arms[f"fp8_cb@{rung}"] = {
            "weighted_snr_db": fixed + rung / 100,
            "footprint": {"exact_bpw": fixed_bpw + rung / 1000},
        }
        arms[f"fp8_cb_learned@{rung}"] = {
            "weighted_snr_db": learned + rung / 100,
            "footprint": {"exact_bpw": learned_bpw + rung / 1000},
        }
    return {"population": population, "arms": arms}


def test_population_summary_never_pools_dense_and_routed():
    summaries = _DRIVER.population_summaries({
        "dense-a": _cell("dense", 10.0, 11.0),
        "dense-b": _cell("dense", 12.0, 13.0),
        "routed-a": _cell("routed", 20.0, 24.0),
    })
    assert set(summaries) == {"dense", "routed"}
    assert summaries["dense"]["tensors"] == 2
    assert summaries["routed"]["tensors"] == 1
    assert all(row["learned_minus_fixed_db_median"] == 1.0
               for row in summaries["dense"]["rows"])
    assert all(row["learned_minus_fixed_db_median"] == 4.0
               for row in summaries["routed"]["rows"])
    assert "all" not in summaries and "pooled" not in summaries
