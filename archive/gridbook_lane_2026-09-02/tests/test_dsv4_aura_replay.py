from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from prismaquant.anchored_cost import CandidateSpec, UnitSpec
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
import prismaquant.dsv4_aura_cb_reprice as campaign
from prismaquant.gridbook_runtime_pin import (
    GRIDBOOK_RUNTIME_RELEASE_COMMIT,
    GRIDBOOK_RUNTIME_RELEASE_VERSION,
)


def _runtime_snapshot_env(monkeypatch, root: Path) -> tuple[str, str, str]:
    commit = "a" * 40
    tree = "b" * 40
    closure = "c" * 64
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", commit)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_DIRTY", "0")
    monkeypatch.setenv("PQ_RUNTIME_PRISMAQUANT_ROOT", str(root))
    monkeypatch.setenv("PQ_RUNTIME_PRISMAQUANT_TREE", tree)
    monkeypatch.setenv(
        "PQ_RUNTIME_PRISMAQUANT_CLOSURE_SHA256", closure
    )
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    return commit, tree, closure


def _mock_runtime_identity() -> dict[str, str]:
    return {
        "snapshot": "/runtime-snapshot",
        "commit": "a" * 40,
        "tree": "b" * 40,
        "closure_sha256": "c" * 64,
    }


def _mock_receipt() -> dict[str, object]:
    runtime = _mock_runtime_identity()
    return {
        "receipt_sha256": "d" * 64,
        "producer": {
            key: runtime[key]
            for key in ("commit", "tree", "closure_sha256")
        },
    }


def test_replay_runtime_commit_requires_and_verifies_own_snapshot(
    monkeypatch, tmp_path,
):
    root = tmp_path / "snapshot"
    module_path = root / "prismaquant" / "dsv4_aura_cb_reprice.py"
    verifier_path = root / "tools" / "prismaquant_runtime_snapshot.py"
    module_path.parent.mkdir(parents=True)
    verifier_path.parent.mkdir(parents=True)
    module_path.write_text("# exact replay module fixture\n", encoding="utf-8")
    verifier_path.write_text(
        "def verify_snapshot(snapshot, *, expected_commit, expected_tree, "
        "expected_closure_sha256):\n"
        "    return {\n"
        "        'snapshot': str(snapshot),\n"
        "        'commit': expected_commit,\n"
        "        'tree': expected_tree,\n"
        "        'closure_sha256': expected_closure_sha256,\n"
        "    }\n",
        encoding="utf-8",
    )
    commit, _tree, _closure = _runtime_snapshot_env(
        monkeypatch, root
    )
    monkeypatch.setattr(campaign, "__file__", str(module_path))
    monkeypatch.setattr(
        campaign,
        "sys",
        SimpleNamespace(
            flags=SimpleNamespace(safe_path=True, no_user_site=True),
            dont_write_bytecode=True,
        ),
    )
    assert campaign._release_runtime_commit() == commit


def test_replay_runtime_commit_refuses_caller_assertion_without_snapshot(
    monkeypatch, tmp_path,
):
    _runtime_snapshot_env(monkeypatch, tmp_path / "absent-snapshot")
    monkeypatch.delenv("PQ_RUNTIME_PRISMAQUANT_ROOT")
    monkeypatch.setattr(
        campaign,
        "sys",
        SimpleNamespace(
            flags=SimpleNamespace(safe_path=True, no_user_site=True),
            dont_write_bytecode=True,
        ),
    )
    with pytest.raises(
        campaign.DSv4CampaignError, match="runtime snapshot identity",
    ):
        campaign._release_runtime_commit()


def test_replay_refuses_completion_receipt_from_different_runtime_tree():
    receipt = _mock_receipt()
    receipt["producer"]["tree"] = "e" * 40
    with pytest.raises(
        campaign.DSv4CampaignError, match="producer.tree",
    ):
        campaign._bind_completion_receipt_to_runtime(
            receipt, _mock_runtime_identity()
        )


def _write_minimal_completed_replay(tmp_path: Path):
    from prismaquant.aura_cost import (
        AURA_CHECKPOINT_IDENTITY_SCHEMA,
        AURA_CHECKPOINT_MANIFEST_SCHEMA,
        AURA_CHECKPOINT_UNIT_SCHEMA,
    )
    from prismaquant.production_weight_cache import (
        _combined_source_weights_sha256,
    )

    qname = "model.layers.0.self_attn.wq_a"
    measured_format = "FP8_CB_K32"
    terminal_format = "FP8_BLOCK_UE8M0_SOURCE"
    unit = UnitSpec(
        qname=qname,
        role="wq_a",
        unit_class="nonexpert",
        n_params=16,
        candidates=(
            CandidateSpec(
                measured_format, 4.0, 8, "fp8_cb", "learned",
                (32.0,), 32.0,
            ),
            CandidateSpec(
                terminal_format, 8.0, 16, "source_terminal",
                "passthrough", (), 0.0,
                terminal=True,
                allocator_selectable=True,
            ),
        ),
    )
    work_dir = tmp_path / "campaign"
    checkpoint_root = work_dir / "checkpoints" / "aura"
    artifact_path = work_dir / "artifacts" / "streamed_anchor_aura.pkl"
    args = SimpleNamespace(
        work_dir=str(work_dir),
        checkpoint_dir=str(work_dir / "checkpoints"),
        n_probes=2,
    )
    arm_identity = {"test_arm": "production"}
    purposes = {qname: {measured_format: ["anchor"]}}
    prepared = SimpleNamespace(
        args=args,
        units=(unit,),
        probe_stats={qname: {
            "out_features": 4,
            "in_features": 4,
            "n_params": 16,
        }},
        probe_meta={"calib_hash": "calibration-hash"},
        purposes_by_qname=purposes,
        formats_by_qname={qname: (terminal_format, measured_format)},
        format_plan=SimpleNamespace(identity_sha256="f" * 64),
        measurement_format_plan_identity_sha256="f" * 64,
        routed_selection_sha256="e" * 64,
        arm_identity=arm_identity,
    )

    base_cb = {
        "schema": "test.cb.render.v1",
        "cb_formats_by_qname": {qname: [measured_format]},
        "immutable_render_input": "bound",
    }
    base_renderer = {
        "schema": "test.production.renderer.v1",
        "arm_identity": arm_identity,
        "formats_by_qname": {qname: [measured_format]},
        "cb_render_identity": base_cb,
        "retention": "one_layer_in_memory",
    }
    source_record = {"shape": [4, 4], "sha256": "a" * 64}
    source_records = {qname: source_record}
    cb_shapes = {qname: [4, 4]}
    cb_content = {qname: "a" * 64}
    completed_cb = {
        **base_cb,
        "source_weights_complete": True,
        "source_weights_shapes": cb_shapes,
        "source_weights_content_sha256": cb_content,
        "source_weights_sha256": _combined_source_weights_sha256(
            cb_shapes, cb_content,
        ),
        "render_scope": "sparse_production_anchors",
    }
    completed_renderer = {
        **base_renderer,
        "cb_render_identity": completed_cb,
        "source_weights": {
            "complete": True,
            "scope": "sparse_anchor_plan",
            "records": source_records,
            "identity_sha256": canonical_json_sha256(
                source_records, where="test source records",
            ),
        },
    }

    state_row = {
        "s2": 13.0,
        "s4": 97.0,
        "x2_probe": [4.0, 9.0],
        "dw_src": "production_render",
    }
    state = {
        "g_trace": 6.0,
        "rows": {measured_format: state_row},
        "col_energy": None,
        "source_weight_identity": source_record,
    }
    measured_row = campaign._expected_checkpoint_cost_row(
        qname,
        measured_format,
        state_row,
        n_probes=2,
        diagnostic_expected=False,
    )
    legacy_zero = {
        "predicted_dloss": 0.0,
        "output_mse_measured": False,
        "cost_source": "aura_passthrough_zero",
    }
    raw_payload = {
        "schema": "prismaquant.aura_cost.v1",
        "n_probes": 2,
        "formats": [measured_format, terminal_format, "MXFP4_SOURCE"],
        "stats": {qname: {
            "h_trace": 3.0,
            "n_params": 16,
            "in_features": 4,
            "out_features": 4,
            "n_probes": 2,
        }},
        "costs": {qname: {
            measured_format: measured_row,
            # Exact historical rows: block-FP8 is now activation-unmeasured;
            # MXFP4 is the old global-zero cross-pollution on this dense unit.
            terminal_format: legacy_zero,
            "MXFP4_SOURCE": legacy_zero,
        }},
        "provenance": {
            "calib_hash": "calibration-hash",
            "production_anchor_render_purposes": purposes,
            "production_anchor_renderer": completed_renderer,
            "production_anchor_sparse_render_identity": completed_cb,
            "cb_render_identity": completed_cb,
            "dw_production_anchor_rows": 1,
        },
    }

    identity = {
        "schema": AURA_CHECKPOINT_IDENTITY_SCHEMA,
        "calibration": {"calib_hash": "calibration-hash"},
        "units": [{
            "qname": qname,
            "shape": [4, 4],
            "dtype": "torch.bfloat16",
            "n_params": 16,
        }],
        "chunks": [[qname]],
        "n_probes": 2,
        "collect_col_energy": False,
        "require_production_cache": True,
        "extra": {
            "campaign_schema": campaign.DSV4_CAMPAIGN_SCHEMA,
            "source_format_plan_identity_sha256": "f" * 64,
            "routed_book_selection_sha256": "e" * 64,
            "include_routed_experts": True,
            "production_anchor_render_purposes": purposes,
            "production_anchor_renderer": base_renderer,
            # The live campaign was launched before the global-zero plan bug
            # was fixed, so both legacy terminal names appear in this identity.
            "streamed_formats_by_qname": {qname: [
                terminal_format, measured_format, "MXFP4_SOURCE",
            ]},
        },
    }
    identity_sha256 = canonical_json_sha256(
        identity, where="test AURA checkpoint identity",
    )
    unit_payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {
        "schema": AURA_CHECKPOINT_UNIT_SCHEMA,
        "qname": qname,
        "identity_sha256": identity_sha256,
        "payload_sha256": hashlib.sha256(unit_payload).hexdigest(),
        "payload": unit_payload,
    }
    unit_name = hashlib.sha256(qname.encode()).hexdigest() + ".pkl"
    manifest = {
        "schema": AURA_CHECKPOINT_MANIFEST_SCHEMA,
        "identity_sha256": identity_sha256,
        "identity": identity,
        "units": [{"qname": qname, "file": f"units/{unit_name}"}],
    }

    (checkpoint_root / "units").mkdir(parents=True)
    (checkpoint_root / "units" / unit_name).write_bytes(
        pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    )
    (checkpoint_root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True)
    )
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(
        pickle.dumps(raw_payload, protocol=pickle.HIGHEST_PROTOCOL)
    )
    return prepared, artifact_path, raw_payload, qname, measured_format


def test_cpu_replay_readmits_w8a16_terminal_and_quarantines_cross_terminal(tmp_path):
    prepared, artifact, _raw, qname, measured = (
        _write_minimal_completed_replay(tmp_path)
    )
    sanitized, provenance = (
        campaign._load_and_audit_completed_streamed_payload(
            prepared, artifact,
        )
    )

    assert set(sanitized["costs"][qname]) == {
        measured, "FP8_BLOCK_UE8M0_SOURCE"
    }
    assert provenance["measurement_invoked"] is False
    assert provenance["legacy_fp8_terminal_zero_rows_quarantined"] == 0
    assert provenance["legacy_fp8_terminal_zero_rows_readmitted"] == 1
    assert provenance["legacy_cross_terminal_zero_rows_quarantined"] == 1
    assert provenance["unit_checkpoint_count"] == 1
    assert provenance["fp8_block_terminal_allocator_selectable"] is True


def test_cpu_replay_refuses_monolithic_scalar_tamper(tmp_path):
    prepared, artifact, raw, qname, measured = (
        _write_minimal_completed_replay(tmp_path)
    )
    raw["costs"][qname][measured]["predicted_dloss"] += 1.0
    artifact.write_bytes(pickle.dumps(raw, protocol=pickle.HIGHEST_PROTOCOL))

    with pytest.raises(
        campaign.DSv4CampaignError, match="scalar differs from journal",
    ):
        campaign._load_and_audit_completed_streamed_payload(
            prepared, artifact,
        )


def test_replay_control_plane_never_invokes_measurement(monkeypatch, tmp_path):
    args = SimpleNamespace(
        replay_streamed_payload=str(tmp_path / "streamed_anchor_aura.pkl"),
        work_dir=str(tmp_path / "work"),
    )
    prepared = SimpleNamespace(args=args)
    receipt_path = (
        Path(args.work_dir) / "artifacts" / "campaign_completion_receipt.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text("{}")
    def prepare(_args, *, publish_format_plan=True):
        assert publish_format_plan is False
        return prepared

    monkeypatch.setattr(campaign, "prepare_dsv4_campaign", prepare)
    monkeypatch.setattr(
        campaign, "require_allocator_supersurrogate_support", lambda: None,
    )
    monkeypatch.setattr(
        campaign, "_release_runtime_identity", _mock_runtime_identity,
    )
    monkeypatch.setattr(
        campaign,
        "verify_receipt_for_replay",
        lambda *_args, **_kwargs: _mock_receipt(),
    )
    monkeypatch.setattr(
        campaign,
        "assert_replay_matches_completion_receipt",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        campaign,
        "_load_and_audit_completed_streamed_payload",
        lambda _prepared, _path: ({"costs": {}}, {"measurement_invoked": False}),
    )

    def forbidden(_prepared):
        raise AssertionError("GPU measurement must not run during replay")

    monkeypatch.setattr(campaign, "_measure_streamed", forbidden)
    observed = {}

    def finish(_prepared, _payload, **kwargs):
        observed.update(kwargs)
        return 0

    monkeypatch.setattr(campaign, "_finish_dsv4_campaign", finish)
    assert campaign.run_dsv4_anchor_replay(
        args, control_plane=campaign.__name__,
    ) == 0
    assert observed["replay_provenance"]["measurement_invoked"] is False
    assert observed["allocator_output"].name == (
        "allocator-aura-activation-safe"
    )
    assert observed["replay_provenance"][
        "campaign_completion_receipt"
    ]["receipt_sha256"] == "d" * 64


def test_replay_refuses_before_prepare_without_terminal_receipt(
    monkeypatch, tmp_path,
):
    args = SimpleNamespace(
        replay_streamed_payload=str(tmp_path / "streamed_anchor_aura.pkl"),
        work_dir=str(tmp_path / "work"),
    )
    monkeypatch.setattr(
        campaign, "_release_runtime_identity", _mock_runtime_identity,
    )

    def refuse(*_args, **_kwargs):
        raise campaign.CampaignCompletionError("receipt absent")

    monkeypatch.setattr(campaign, "verify_receipt_for_replay", refuse)
    monkeypatch.setattr(
        campaign,
        "prepare_dsv4_campaign",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepare must not run before receipt admission")
        ),
    )
    with pytest.raises(
        campaign.DSv4CampaignError, match="valid terminal campaign receipt",
    ):
        campaign.run_dsv4_anchor_replay(
            args, control_plane=campaign.__name__,
        )


def test_replay_refuses_receipt_mismatch_after_deep_audit(monkeypatch, tmp_path):
    args = SimpleNamespace(
        replay_streamed_payload=str(tmp_path / "streamed_anchor_aura.pkl"),
        work_dir=str(tmp_path / "work"),
    )
    receipt_path = (
        Path(args.work_dir) / "artifacts" / "campaign_completion_receipt.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text("{}")
    monkeypatch.setattr(
        campaign, "_release_runtime_identity", _mock_runtime_identity,
    )
    monkeypatch.setattr(
        campaign,
        "verify_receipt_for_replay",
        lambda *_args, **_kwargs: _mock_receipt(),
    )
    monkeypatch.setattr(
        campaign, "prepare_dsv4_campaign",
        lambda *_args, **_kwargs: SimpleNamespace(args=args),
    )
    monkeypatch.setattr(
        campaign, "require_allocator_supersurrogate_support", lambda: None,
    )
    monkeypatch.setattr(
        campaign, "_load_and_audit_completed_streamed_payload",
        lambda *_args, **_kwargs: ({"costs": {}}, {"measurement_invoked": False}),
    )
    monkeypatch.setattr(
        campaign, "assert_replay_matches_completion_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            campaign.CampaignCompletionError("payload set differs")
        ),
    )
    with pytest.raises(
        campaign.DSv4CampaignError, match="differs from terminal receipt",
    ):
        campaign.run_dsv4_anchor_replay(
            args, control_plane=campaign.__name__,
        )


def _format_plan_fixture(*, runtime_version: str, rungs_source: str):
    return {
        "schema": "prismaquant.source_class_format_plan.v1",
        "menus": {"expert": ["FP8_CB_K28"], "nonexpert": ["FP8_CB_K28"]},
        "units": {"fixture": {"source_kind": "fp8_ue8m0"}},
        "serving_groups": [],
        "serving_backed_restriction": {
            "profile_id": "nvfp4_cb",
            "family": "fp8_cb",
            "runtime_version": runtime_version,
            "rungs_source": rungs_source,
            "fused_mid_m_rungs": [28, 32, 36, 40, 44, 48],
        },
    }


def test_w8a16_format_plan_migration_allows_only_runtime_provenance():
    historical = _format_plan_fixture(
        runtime_version="0.8.4", rungs_source="serving_profile_spec:0.8.4"
    )
    historical["identity_sha256"] = (
        campaign.DSV4_W8A16_LEGACY_FORMAT_PLAN_SHA256
    )
    pinned = campaign.GRIDBOOK_RUNTIME_RELEASE_VERSION
    current = _format_plan_fixture(
        runtime_version=pinned,
        rungs_source=f"serving_profile_spec:{pinned}",
    )
    current["identity_sha256"] = "a" * 64
    old = SimpleNamespace(to_dict=lambda: historical)
    new = SimpleNamespace(to_dict=lambda: current)

    proof = campaign._w8a16_format_plan_delta(old, new)
    assert proof["semantic_payload_equal_after_allowed_delta"] is True
    assert proof["historical_identity_sha256"] == (
        campaign.DSV4_W8A16_LEGACY_FORMAT_PLAN_SHA256
    )

    changed = json.loads(json.dumps(current))
    changed["menus"]["nonexpert"].append("FP8_CB_K32")
    with pytest.raises(campaign.DSv4CampaignError, match="changed beyond"):
        campaign._w8a16_format_plan_delta(
            old, SimpleNamespace(to_dict=lambda: changed)
        )


def test_w8a16_readmission_release_pin_is_exact_and_released():
    proof = campaign._w8a16_runtime_contract_proof()
    assert proof["commit"] == GRIDBOOK_RUNTIME_RELEASE_COMMIT
    assert proof["version"] == GRIDBOOK_RUNTIME_RELEASE_VERSION
    assert proof["version_is_release"] is True
    assert proof["required_abi_features"][
        "source_fp8_block128_w8a16"
    ] == 1


def test_w8a16_legacy_receipt_allowlist_is_exact():
    receipt = {
        "receipt_sha256": campaign.DSV4_W8A16_LEGACY_RECEIPT_SHA256,
        "producer": dict(campaign.DSV4_W8A16_LEGACY_PRODUCER),
        "service": {
            "result": "success",
            "invocation_id": campaign.DSV4_W8A16_LEGACY_INVOCATION_ID,
        },
    }
    campaign._assert_w8a16_legacy_receipt(receipt)
    receipt["service"]["invocation_id"] = "wrong"
    with pytest.raises(campaign.DSv4CampaignError, match="not allowlisted"):
        campaign._assert_w8a16_legacy_receipt(receipt)
