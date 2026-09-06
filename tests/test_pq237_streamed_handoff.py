"""Actual streamed rows cross a pickle/checkpoint/candidate boundary on CPU."""
import copy
import pickle

import pytest

from experiments.pq237_joint_aura_streamed import compare_assignments, load_candidate_payload
from prismaquant.joint_aura import identity_sha256, make_joint_aura_entry
from test_joint_aura_streamed import _fixture, _run


def _write(path, payload):
    path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def test_streamed_checkpoint_pickle_candidates_and_identity_refusal(tmp_path, monkeypatch):
    import prismaquant.aura_cost as aura

    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "0")
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    _, _, runner, cache = _fixture()
    first = _run(runner, cache, checkpoint_dir=tmp_path / "checkpoints")
    _, context, runner, cache = _fixture()
    restored = _run(runner, cache, checkpoint_dir=tmp_path / "checkpoints", resume=True)
    assert context.install_calls == 0
    assert restored["costs"] == first["costs"]
    plan = {name: tuple(rows) for name, rows in first["costs"].items()}
    bindings = {name: {fmt: row["joint_operator_identity_sha256"] for fmt, row in rows.items()}
                for name, rows in first["costs"].items()}
    probe = first["provenance"]["probe_identity_sha256"]
    path = tmp_path / "joint.pkl"
    _write(path, restored)
    payload, candidates, receipt = load_candidate_payload(path, plan, bindings, probe)
    assert receipt["currency"]["joint_aura_rows"] == sum(map(len, plan.values()))
    for name, rows in candidates.items():
        assert {row.fmt for row in rows} == set(plan[name])
        for candidate in rows:
            assert candidate.predicted_dloss == payload["costs"][name][candidate.fmt]["predicted_dloss"]

    changed = copy.deepcopy(restored)
    name = next(iter(plan))
    changed["costs"][name].pop("BF16")
    _write(path, changed)
    with pytest.raises(ValueError, match="coordinate scope"):
        load_candidate_payload(path, plan, bindings, probe)

    changed = copy.deepcopy(restored)
    changed["stats"][name]["n_params"] += 1
    _write(path, changed)
    with pytest.raises(ValueError, match="bound dense source shape"):
        load_candidate_payload(path, plan, bindings, probe)

    _write(path, restored)
    wrong = copy.deepcopy(bindings)
    wrong[name]["BF16"] = "f" * 64
    with pytest.raises(ValueError, match="operator binding"):
        load_candidate_payload(path, plan, wrong, probe)
    with pytest.raises(ValueError, match="probe identity"):
        load_candidate_payload(path, plan, bindings, "e" * 64)

    # A internally consistent producer payload is still wrong if it changes
    # the predeclared common probes. Rehash every affected row and envelope.
    changed = copy.deepcopy(restored)
    wrong_probe = copy.deepcopy(changed["provenance"]["probe_identity"])
    wrong_probe["seed_base"] += 100
    wrong_digest = identity_sha256(wrong_probe)
    for rows in changed["costs"].values():
        for fmt, row in list(rows.items()):
            operator = copy.deepcopy(row["joint_operator_identity"])
            operator["probe_identity_sha256"] = wrong_digest
            rows[fmt] = make_joint_aura_entry(operator_identity=operator, probe_identity=wrong_probe,
                                            signed_components=row["signed_components_per_probe"])
    provenance = changed["provenance"]
    provenance["probe_identity"] = wrong_probe
    provenance["probe_identity_sha256"] = wrong_digest
    provenance["joint_aura_identity"]["probe_identity"] = wrong_probe
    provenance["joint_aura_identity_sha256"] = identity_sha256(provenance["joint_aura_identity"])
    _write(path, changed)
    with pytest.raises(ValueError, match="probe identity"):
        load_candidate_payload(path, plan, bindings, probe)

    changed = copy.deepcopy(restored)
    changed["provenance"]["joint_aura_identity"] = copy.deepcopy(
        changed["provenance"]["joint_aura_identity"])
    changed["provenance"]["joint_aura_identity"]["probe_identity"]["temperature"] = 2.0
    _write(path, changed)
    with pytest.raises(ValueError, match="provenance"):
        load_candidate_payload(path, plan, bindings, probe)


def test_four_assignment_diagnostics_keep_background_and_currency_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "0")
    _, _, runner, cache = _fixture()
    payload = _run(runner, cache)
    plan = {name: tuple(rows) for name, rows in payload["costs"].items()}
    bindings = {name: {fmt: row["joint_operator_identity_sha256"] for fmt, row in rows.items()}
                for name, rows in payload["costs"].items()}
    path = tmp_path / "joint.pkl"
    _write(path, payload)
    payload, candidates, _ = load_candidate_payload(path, plan, bindings,
                                                    payload["provenance"]["probe_identity_sha256"])
    background, swapped = list(plan)
    assignments = {
        f"L0{left}_L21{right}": {background: fmt_left, swapped: fmt_right}
        for left, fmt_left in (("A8", "FP8_E4M3"), ("A16", "BF16"))
        for right, fmt_right in (("A4", "NVFP4A16"), ("A16", "BF16"))
    }
    scores, pairs = compare_assignments(payload, candidates, assignments)
    for score in scores.values():
        assert score["allocator_mean_cost"] == pytest.approx(score["joint_additive"]["mean"])
        assert score["joint_quadratic_diagnostic"]["objective"] == "joint_quadratic"
        assert "not a joint allocator row" in score["weight_component_additive_diagnostic"]["role"]
    assert pairs["A8"]["joint_additive"]["difference_per_probe"] == pytest.approx(
        pairs["A16"]["joint_additive"]["difference_per_probe"])
    assert pairs["A8"]["joint_quadratic_diagnostic"]["difference_per_probe"] != pytest.approx(
        pairs["A16"]["joint_quadratic_diagnostic"]["difference_per_probe"], abs=1e-12)
    assert set(pairs["A8"]["joint_quadratic_diagnostic"]["assignment_a"]
               ["operator_identity_sha256_by_unit"]) == set(plan)
