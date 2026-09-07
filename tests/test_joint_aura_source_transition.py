"""CPU gates for explicit adoption; no mutation of external checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import pickle

import pytest

from prismaquant import aura_cost as aura
from prismaquant import joint_aura_source_transition as transition
from test_joint_aura_streamed import _fixture, _run


def _write_json(path, value):
    path.write_bytes(transition._canonical(value) + b"\n")
    return {"path": str(path), "sha256": transition._sha(path)}


def _build_adopted(tmp_path, monkeypatch, *, layers=2):
    old_git, old_source, new_git, new_source = "1" * 40, "2" * 64, "3" * 40, "4" * 64
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: old_git)
    monkeypatch.setattr(aura, "_aura_source_sha256", lambda: old_source)
    root = tmp_path / "checkpoints"
    _, _, runner, cache = _fixture()
    expected = _run(runner, cache, checkpoint_dir=root)
    manifest = json.loads((root / "manifest.json").read_bytes())
    pending = aura._aura_unit_checkpoint_path(root, "model.layers.0.proj")
    for i in range(layers - 1):
        aura._aura_unit_checkpoint_path(root, f"model.layers.{i}.proj").unlink()
    kept_name = f"model.layers.{layers - 1}.proj"
    kept = aura._aura_unit_checkpoint_path(root, kept_name)
    kept_bytes = kept.read_bytes()
    envelope = pickle.loads(kept_bytes)
    preserved = [{"qname": kept_name, "file": kept.relative_to(root).as_posix(),
                  "sha256": transition._sha(kept), "bytes": len(kept_bytes),
                  "payload_sha256": envelope["payload_sha256"]}]
    config = {"output_root": str(tmp_path), "calibration": "fixed", "backend": "fixed"}
    plan = _write_json(tmp_path / "plan.json", config)
    cache_file = tmp_path / "production.pkl"
    cache_file.write_bytes(b"prepared test fixture")
    prepared = _write_json(tmp_path / "prepared.json", {
        "implementation_sha256": old_source, "plan_sha256": plan["sha256"],
        "production_cache": {"path": str(cache_file), "sha256": transition._sha(cache_file)}})
    inspection = _write_json(tmp_path / "inspection.json", {
        "identity_sha256": manifest["identity_sha256"], "manifest_sha256": transition._sha(root / "manifest.json"),
        "original_source_commit": old_git, "completed_units": 1, "total_units": layers, "units": preserved})
    contract = {"source_sha256": old_source, "git_commit": old_git,
        "manifest_sha256": transition._sha(root / "manifest.json"), "identity_sha256": manifest["identity_sha256"],
        "inspection_sha256": inspection["sha256"], "plan_sha256": plan["sha256"],
        "prepared_sha256": prepared["sha256"], "production_cache_sha256": transition._sha(cache_file),
        "preserved_units": 1, "total_units": layers}
    execution = {"git_commit": new_git, "producer_source_sha256": new_source,
                 "reconstructed_source_sha256": old_source, "transition_module_sha256": "5" * 64}
    monkeypatch.setattr(transition, "_CONTRACT", contract)
    monkeypatch.setattr(transition, "_actual_execution", lambda: dict(execution))
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: execution["git_commit"])
    monkeypatch.setattr(aura, "_aura_source_sha256", lambda: new_source)
    bindings = {"plan": plan, "prepared": prepared, "inspection": inspection}
    receipt = transition.create_transition(bindings=bindings, checkpoint_dir=root, output=tmp_path / "transition.json")
    def load():
        return transition.load_transition(receipt, config=config, plan_sha256=plan["sha256"],
                                          prepared=prepared, checkpoint_dir=root)
    return {"root": root, "kept": kept, "kept_bytes": kept_bytes, "pending": pending,
            "expected": expected, "load": load, "receipt": receipt, "execution": execution,
            "config": config, "bindings": bindings, "manifest": manifest}


@pytest.fixture
def adopted(tmp_path, monkeypatch):
    return _build_adopted(tmp_path, monkeypatch)


@pytest.fixture
def adopted_three(tmp_path, monkeypatch):
    from test_streamed_cost_checkpoints import _DenseTinyLM, _DenseLayer
    original_init = _DenseTinyLM.__init__
    def init(self, state=None, **kwargs):
        original_init(self, None, **kwargs)
        self.model.layers.append(_DenseLayer())
        if state is not None:
            self.load_state_dict(state)
    monkeypatch.setattr(_DenseTinyLM, "__init__", init)
    return _build_adopted(tmp_path, monkeypatch, layers=3)


@pytest.mark.parametrize("mutation", ["other_file", "math", "glue", "missing_module", "extra_file"])
def test_source_proof_rejects_every_unapproved_change(tmp_path, monkeypatch, mutation):
    # Tiny byte tree exercises the exact algorithm independently of repo size.
    (tmp_path / "aura_cost.py").write_bytes(b"new-math\nnew-glue\n")
    (tmp_path / "dependency.py").write_bytes(b"unchanged\n")
    (tmp_path / "joint_aura_source_transition.py").write_bytes(b"verifier\n")
    monkeypatch.setattr(transition, "_SOURCE_REWRITES", {"aura_cost.py": [("old-math", "new-math"), ("old-glue", "new-glue")]})
    from prismaquant.production_weight_cache import _production_cache_source_sha256
    old = tmp_path / "original"
    old.mkdir()
    (old / "aura_cost.py").write_bytes(b"old-math\nold-glue\n")
    (old / "dependency.py").write_bytes(b"unchanged\n")
    contract = dict(transition._CONTRACT, source_sha256=_production_cache_source_sha256(old))
    for path in old.iterdir():
        path.unlink()
    old.rmdir()
    monkeypatch.setattr(transition, "_CONTRACT", contract)
    transition.source_proof(tmp_path)
    if mutation == "other_file":
        (tmp_path / "dependency.py").write_bytes(b"changed\n")
    elif mutation == "math":
        (tmp_path / "aura_cost.py").write_bytes(b"wrong-math\nnew-glue\n")
    elif mutation == "glue":
        (tmp_path / "aura_cost.py").write_bytes(b"new-math\nwrong-glue\n")
    elif mutation == "missing_module":
        (tmp_path / "joint_aura_source_transition.py").unlink()
    else:
        (tmp_path / "extra.py").write_bytes(b"new dependency")
    with pytest.raises(ValueError, match="source|package"):
        transition.source_proof(tmp_path)


def test_real_producer_resumes_exact_rows_and_records_new_execution(adopted):
    cap = adopted["load"]()
    _, _, runner, cache = _fixture()
    resumed = _run(runner, cache, checkpoint_dir=adopted["root"], resume=True, source_transition=cap)
    assert resumed["costs"] == adopted["expected"]["costs"]
    assert adopted["kept"].read_bytes() == adopted["kept_bytes"]
    state = aura._load_aura_unit_checkpoint(adopted["pending"], qname="model.layers.0.proj",
                                           identity_sha256=adopted["manifest"]["identity_sha256"])
    assert state["execution_provenance"] == cap.execution_provenance
    provenance = resumed["provenance"]["source_transition"]
    assert (provenance["preserved_count"], provenance["new_count"]) == (1, 1)
    assert provenance["execution"] == adopted["execution"]
    # A second interruption/restart admits only new units carrying this receipt.
    cap2 = adopted["load"]()
    _, context, runner, cache = _fixture()
    again = _run(runner, cache, checkpoint_dir=adopted["root"], resume=True, source_transition=cap2)
    assert again["costs"] == resumed["costs"]
    assert context.install_calls == 0


@pytest.mark.parametrize("target", ["kept", "manifest", "plan", "prepared", "inspection", "cache", "receipt"])
def test_preflight_rejects_changed_bound_artifact(adopted, target):
    if target == "kept":
        path = adopted["kept"]
    elif target == "manifest":
        path = adopted["root"] / "manifest.json"
    elif target == "cache":
        path = adopted["root"].parent / "production.pkl"
    elif target == "receipt":
        path = Path(adopted["receipt"]["path"])
    else:
        path = Path(adopted["bindings"][target]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        adopted["load"]()


def test_preflight_rejects_resealed_receipt_with_other_source(adopted):
    path = Path(adopted["receipt"]["path"])
    record = json.loads(path.read_bytes())
    record["execution"]["producer_source_sha256"] = "9" * 64
    adopted["receipt"].update(_write_json(path, record))
    with pytest.raises(ValueError, match="execution source"):
        adopted["load"]()


def test_preflight_rejects_changed_runtime_plan(adopted):
    adopted["config"]["backend"] = "different"
    with pytest.raises(ValueError, match="runtime plan"):
        adopted["load"]()


def test_resume_rejects_changed_non_source_identity(adopted):
    cap = adopted["load"]()
    _, context, runner, cache = _fixture()
    with pytest.raises(ValueError, match="non-source checkpoint identity"):
        _run(runner, cache, checkpoint_dir=adopted["root"], resume=True,
             source_transition=cap, seed_base=7001)
    assert context.install_calls == 0
    assert not adopted["pending"].exists()


def test_preflight_rejects_unbound_new_unit(adopted):
    source = pickle.loads(adopted["kept_bytes"])
    state = pickle.loads(source["payload"])
    aura._write_aura_unit_checkpoint(adopted["root"], qname="model.layers.0.proj",
        identity_sha256=adopted["manifest"]["identity_sha256"], state=state)
    with pytest.raises(ValueError, match="new unit lacks bound execution provenance"):
        adopted["load"]()


@pytest.mark.parametrize("kind", ["dict", "forged"])
def test_caller_cannot_supply_identity_override(adopted, kind):
    cap = adopted["load"]()
    bad = cap.execution_provenance if kind == "dict" else transition.VerifiedTransition(
        cap._receipt_bytes, cap._receipt_path, cap._receipt_sha256, cap._checkpoint_dir, cap._manifest_bytes)
    _, context, runner, cache = _fixture()
    with pytest.raises(ValueError, match="verified receipt loader"):
        _run(runner, cache, checkpoint_dir=adopted["root"], resume=True, source_transition=bad)
    assert context.install_calls == 0


def test_source_changed_after_preflight_rejected(adopted):
    cap = adopted["load"]()
    adopted["execution"]["transition_module_sha256"] = "0" * 64
    _, context, runner, cache = _fixture()
    with pytest.raises(ValueError, match="source changed after admission"):
        _run(runner, cache, checkpoint_dir=adopted["root"], resume=True, source_transition=cap)
    assert context.install_calls == 0


def test_transition_receipt_cannot_be_overwritten(adopted):
    with pytest.raises(FileExistsError):
        transition.create_transition(bindings=adopted["bindings"], checkpoint_dir=adopted["root"],
                                     output=adopted["receipt"]["path"])


def test_execute_preflight_precedes_cuda_and_model_work():
    from prismaquant.tessera_joint_aura import execute
    with pytest.raises(ValueError, match="joint source transition"):
        execute("run", {"output_root": "/does/not/exist"}, plan_sha256="0" * 64,
                resume=True, source_transition={"path": "/does/not/exist", "sha256": "0" * 64})


def test_legacy_git_override_is_not_accepted(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "1" * 40)
    with pytest.raises(ValueError, match="legacy Git identity override"):
        transition._actual_execution()


def test_two_interruptions_different_pb_heads_preserve_exact_costs(adopted_three, monkeypatch):
    data = adopted_three
    cap = data["load"]()
    writer = aura._write_aura_unit_checkpoint
    def stop_after_middle(*args, **kwargs):
        writer(*args, **kwargs)
        if kwargs["qname"] == "model.layers.1.proj":
            raise TimeoutError("second admitted action reached deadline")
    monkeypatch.setattr(aura, "_write_aura_unit_checkpoint", stop_after_middle)
    _, _, runner, cache = _fixture()
    with pytest.raises(TimeoutError, match="deadline"):
        _run(runner, cache, checkpoint_dir=data["root"], resume=True, source_transition=cap)
    middle = aura._aura_unit_checkpoint_path(data["root"], "model.layers.1.proj")
    middle_bytes = middle.read_bytes()
    assert not data["pending"].exists()
    previous = dict(data["receipt"])
    data["execution"]["git_commit"] = "6" * 40
    with pytest.raises(ValueError, match="actual execution source differs"):
        data["load"]()
    second = transition.create_transition(bindings=data["bindings"], checkpoint_dir=data["root"],
        output=data["root"].parent / "transition-next.json", predecessor=previous)
    data["receipt"].update(second)
    next_cap = data["load"]()
    monkeypatch.setattr(aura, "_write_aura_unit_checkpoint", writer)
    _, _, runner, cache = _fixture()
    result = _run(runner, cache, checkpoint_dir=data["root"], resume=True, source_transition=next_cap)
    assert result["costs"] == data["expected"]["costs"]
    assert data["kept"].read_bytes() == data["kept_bytes"]
    assert middle.read_bytes() == middle_bytes
    rows = result["provenance"]["source_transition"]["new_units"]
    assert {row["execution_provenance"]["execution"]["git_commit"] for row in rows} == {"3" * 40, "6" * 40}
    assert {row["execution_provenance"]["receipt"]["sha256"] for row in rows} == {previous["sha256"], second["sha256"]}
    # A third PB snapshot can adopt the complete chain without rewriting rows.
    data["execution"]["git_commit"] = "7" * 40
    third = transition.create_transition(bindings=data["bindings"], checkpoint_dir=data["root"],
        output=data["root"].parent / "transition-third.json", predecessor=dict(second))
    data["receipt"].update(third)
    final_cap = data["load"]()
    _, context, runner, cache = _fixture()
    final = _run(runner, cache, checkpoint_dir=data["root"], resume=True, source_transition=final_cap)
    assert final["costs"] == data["expected"]["costs"]
    assert context.install_calls == 0


def test_predecessor_rejects_changed_package(adopted):
    adopted["execution"]["git_commit"] = "6" * 40
    adopted["execution"]["producer_source_sha256"] = "7" * 64
    with pytest.raises(ValueError, match="predecessor execution source differs"):
        transition.create_transition(bindings=adopted["bindings"], checkpoint_dir=adopted["root"],
            output=adopted["root"].parent / "changed-package.json", predecessor=adopted["receipt"])


def test_predecessor_cycle_guard(adopted):
    with pytest.raises(ValueError, match="cyclic"):
        transition._read_chain(adopted["receipt"], execution=adopted["execution"],
                               seen=frozenset({adopted["receipt"]["sha256"]}))


def test_new_receipt_requires_predecessor_for_previously_written_units(adopted):
    cap = adopted["load"]()
    _, _, runner, cache = _fixture()
    _run(runner, cache, checkpoint_dir=adopted["root"], resume=True, source_transition=cap)
    with pytest.raises(ValueError, match="new unit lacks bound execution"):
        transition.create_transition(bindings=adopted["bindings"], checkpoint_dir=adopted["root"],
                                     output=adopted["root"].parent / "missing-predecessor.json")


@pytest.mark.parametrize("mutation", ["prior_bytes", "unit_bytes", "fabricated_execution", "unbound_predecessor"])
def test_predecessor_chain_rejects_tampering(adopted, mutation):
    cap = adopted["load"]()
    _, _, runner, cache = _fixture()
    _run(runner, cache, checkpoint_dir=adopted["root"], resume=True, source_transition=cap)
    previous = dict(adopted["receipt"])
    adopted["execution"]["git_commit"] = "6" * 40
    second = transition.create_transition(bindings=adopted["bindings"], checkpoint_dir=adopted["root"],
        output=adopted["root"].parent / "transition-second.json", predecessor=previous)
    adopted["receipt"].update(second)
    if mutation == "prior_bytes":
        path = Path(previous["path"])
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "unit_bytes":
        adopted["pending"].write_bytes(adopted["pending"].read_bytes() + b" ")
    else:
        path = Path(second["path"])
        record = json.loads(path.read_bytes())
        if mutation == "fabricated_execution":
            record["adopted_units"][0]["execution_provenance"]["execution"]["git_commit"] = "9" * 40
        else:
            record["predecessor"] = None
        adopted["receipt"].update(_write_json(path, record))
    with pytest.raises(ValueError, match="changed|fabricated|require"):
        adopted["load"]()
