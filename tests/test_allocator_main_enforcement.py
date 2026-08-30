"""QC M33 redo: pin allocator main()'s ENFORCEMENT WIRING, not just helpers.

The 2026-06-09 review found the candidate-membership validator tested only
at the helper level — deleting the main() call site left every test green.
This drives the real `allocator.main()` on a synthetic dense model
(probe + cost pickles, DefaultProfile explicitly allowed) with the
validator monkeypatched to a recording spy, asserting main() actually
invokes it on the final expanded assignment. Deleting the call site in
main() fails this test.
"""
from __future__ import annotations

import json
import pickle
import sys

import pytest

import prismaquant.allocator as alloc


def _write_fixture(tmp_path):
    names = [f"model.layers.{i}.self_attn.o_proj" for i in range(4)]
    stats = {
        n: {"h_trace": 1.0 + 0.1 * i, "n_params": 4096 * 4096,
            "shape": [4096, 4096]}
        for i, n in enumerate(names)
    }
    probe = {"stats": stats, "meta": {"model": None}}
    costs = {
        "costs": {
            n: {
                "NVFP4": {"weight_mse": 1e-4 * (i + 1),
                          "output_mse": 1e-4 * (i + 1),
                          "output_mse_measured": True,
                          "predicted_dloss": 1e-4 * (i + 1)},
                "FP8_E4M3": {"weight_mse": 1e-6 * (i + 1),
                             "output_mse": 1e-6 * (i + 1),
                             "output_mse_measured": True,
                             "predicted_dloss": 1e-6 * (i + 1)},
            }
            for i, n in enumerate(names)
        },
        "meta": {"formats": ["NVFP4", "FP8_E4M3"]},
    }
    probe_p = tmp_path / "probe.pkl"
    cost_p = tmp_path / "cost.pkl"
    probe_p.write_bytes(pickle.dumps(probe))
    cost_p.write_bytes(pickle.dumps(costs))
    return probe_p, cost_p


def test_main_invokes_candidate_membership_validator(tmp_path, monkeypatch):
    probe_p, cost_p = _write_fixture(tmp_path)
    lc = tmp_path / "layer_config.json"
    csv = tmp_path / "pareto.csv"

    calls = []
    real = alloc._validate_assignment_candidate_membership

    def spy(assignment, candidates, **kw):
        calls.append(dict(assignment))
        return real(assignment, candidates, **kw)

    monkeypatch.setattr(
        alloc, "_validate_assignment_candidate_membership", spy)
    monkeypatch.setattr(sys, "argv", [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", "NVFP4,FP8_E4M3",
        "--target-bits", "8.0",
        "--pareto-targets", "5.0,8.0",
        "--layer-config", str(lc),
        "--pareto-csv", str(csv),
        "--allow-default-profile",
    ])
    alloc.main()

    assert calls, "main() must invoke the candidate-membership validator"
    assert lc.exists(), "main() must still emit the layer config"
    final = calls[-1]
    emitted = json.loads(lc.read_text())
    emitted_names = set(
        emitted.get("assignment", emitted).keys()
    )
    assert emitted_names & set(final.keys()), (
        "validator must see the assignment that is emitted")


def test_main_refuses_reader_only_fp8_cb_rung_before_allocation(
    tmp_path, monkeypatch
):
    probe_p, cost_p = _write_fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", "FP8_CB_K29,BF16",
        "--target-bits", "8.0",
        "--layer-config", str(tmp_path / "layer_config.json"),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
        "--allow-default-profile",
    ])

    with pytest.raises(SystemExit, match="reader-only.*FP8_CB_K29"):
        alloc.main()
