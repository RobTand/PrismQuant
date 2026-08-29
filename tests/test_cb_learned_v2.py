from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import cb_learned_bundle as bundle
from prismaquant import build_cb_learned_bundle as builder
from prismaquant.cb_imatrix import (
    CB_IMATRIX_FROM_PROBE_SCHEMA,
    canonical_imatrix_sha256,
)
from prismaquant.cb_learned_promotion import (
    CBL_PROMOTION_RECEIPT_SCHEMA,
    CBL_PROMOTION_THRESHOLDS,
    CBL_STEP4_RUNGS,
    CBL_V2_TRAINER_SCHEMA,
    role_census_for_qnames,
)
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA


def _receipt(
    col_weights: dict[str, torch.Tensor],
    *,
    learned_rungs: set[int],
    weights: dict[str, torch.Tensor] | None = None,
    source_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    source_identity = source_identity or _source_identity()
    role_census = role_census_for_qnames(col_weights)
    holdout_ids = ("holdout-a", "holdout-b")
    rungs = {}
    for rung in CBL_STEP4_RUNGS:
        passes = rung in learned_rungs
        metrics = {
            "geomean_ratio": 0.95 if passes else 1.0,
            "bootstrap_95_upper": 0.99,
            "role_aggregate_ratios": {
                role: 1.0 for role in role_census
            },
            "role_coverage_counts": dict(role_census),
            "p95_unit_ratio": 1.02,
            "worst_unit_ratio": 1.08,
        }
        rungs[str(rung)] = {
            "declared_source": "learned" if passes else "lattice",
            "repeat_delta_percentage_points": 0.2,
            "density_shortfall_cells": 0,
            "holdouts": {
                name: copy.deepcopy(metrics) for name in holdout_ids
            },
        }
    candidate_codebooks = {
        qname: {
            str(rung): {
                "subtable_content_sha256": [
                    hashlib.sha256(
                        f"{qname}\0K{rung}\0sub{index}".encode("utf-8")
                    ).hexdigest()
                    for index in range(4)
                ],
            }
            for rung in CBL_STEP4_RUNGS
        }
        for qname in col_weights
    }
    for qname, weight in (weights or {}).items():
        cw = col_weights[qname]
        for rung in learned_rungs:
            _rows, sampling = bundle.learned_v2_sampling_plan(
                qname=qname,
                output_rows=int(weight.shape[-2]),
                in_features=int(weight.shape[-1]),
                population=(int(weight.shape[0]) if weight.ndim == 3 else 1),
                rung=rung,
            )
            if sampling["density_shortfall"] is True or weight.ndim != 2:
                continue
            result = bundle.learn_pool_v2(
                weight.unsqueeze(0),
                cw.reshape(1, int(weight.shape[-1])),
                rung,
                qname=qname,
            )
            candidate_codebooks[qname][str(rung)] = {
                "subtable_content_sha256": [
                    bundle.codebook_table_sha256(table)
                    for table in result.tables
                ],
            }
    return {
        "schema": CBL_PROMOTION_RECEIPT_SCHEMA,
        "model": {
            "model_id": source_identity["source"],
            "content_sha256": source_identity["content_sha256"],
            "role_census": role_census,
        },
        "trainer": {"schema": CBL_V2_TRAINER_SCHEMA},
        "imatrix": {
            "schema": CB_IMATRIX_FROM_PROBE_SCHEMA,
            "calibration_hash": "training-calibration",
            "value_sha256": canonical_imatrix_sha256(col_weights),
        },
        "holdouts": [
            {"id": "holdout-a", "calibration_hash": "b" * 64},
            {"id": "holdout-b", "calibration_hash": "c" * 64},
        ],
        "thresholds": dict(CBL_PROMOTION_THRESHOLDS),
        "rungs": rungs,
        "candidate_codebooks": candidate_codebooks,
    }


def _source_identity(*, source: str = "fixture/model") -> dict[str, object]:
    value_bearing = {
        "config": {"model_type": "fixture"},
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": (
                "model.layers.0.self_attn.q_proj.weight"
            ),
        },
        "shards": [{
            "path": "/fixture/model.safetensors",
            "size": 1,
            "sha256": "d" * 64,
        }],
        "checkpoint_weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
        },
    }
    return {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "source": source,
        "resolved_commit": "fixture-revision",
        "content_sha256": canonical_json_sha256(
            value_bearing,
            where="learned-v2 fixture source identity",
        ),
        **value_bearing,
    }


def _promotion_kwargs(
    col_weights: dict[str, torch.Tensor],
    source_identity: dict[str, object],
) -> dict[str, object]:
    return {
        "source_model_identity": source_identity,
        "probe_calibration_hash": "training-calibration",
        "imatrix_value_sha256": canonical_imatrix_sha256(col_weights),
    }


def _small_inputs(device: str = "cpu"):
    qname = "model.layers.0.self_attn.q_proj"
    weight = torch.linspace(-1.0, 1.0, 64 * 32, device=device).reshape(64, 32)
    col_weights = {qname: torch.linspace(0.5, 1.5, 32, device=device)}
    return qname, weight, col_weights


def test_sampling_plan_uses_density_formula_cap_and_qname_key():
    rows, record = bundle.learned_v2_sampling_plan(
        qname="a.q_proj",
        output_rows=300,
        in_features=256,
        population=1,
        rung=28,
    )
    repeated, repeated_record = bundle.learned_v2_sampling_plan(
        qname="a.q_proj",
        output_rows=300,
        in_features=256,
        population=1,
        rung=28,
    )
    other, other_record = bundle.learned_v2_sampling_plan(
        qname="b.q_proj",
        output_rows=300,
        in_features=256,
        population=1,
        rung=28,
    )
    wider, _ = bundle.learned_v2_sampling_plan(
        qname="a.q_proj",
        output_rows=300,
        in_features=256,
        population=1,
        rung=48,
    )
    assert record["entries"] == 128
    assert record["target_rows"] == 256
    assert record["requested_rows"] == 256
    assert record["sample_rows"] == 256
    assert record["selected_vectors"] == 8192
    assert record["density_shortfall"] is False
    assert torch.equal(rows, repeated)
    assert record["row_selection_sha256"] == repeated_record["row_selection_sha256"]
    assert not torch.equal(rows, other)
    assert record["row_selection_sha256"] != other_record["row_selection_sha256"]
    assert torch.equal(rows, wider[:len(rows)])

    _capped_rows, capped = bundle.learned_v2_sampling_plan(
        qname="experts.down_proj",
        output_rows=512,
        in_features=4096,
        population=10,
        rung=48,
    )
    assert capped["available_vectors"] == 2_621_440
    assert capped["selected_vectors"] == bundle.LLOYD_CAP


def test_sampling_plan_records_density_shortfall_on_small_matrix():
    _rows, record = bundle.learned_v2_sampling_plan(
        qname="small.q_proj",
        output_rows=8,
        in_features=32,
        population=1,
        rung=48,
    )
    assert record["entries"] == 4096
    assert record["target_rows"] == 65536
    assert record["sample_rows"] == 8
    assert record["selected_vectors"] == 32
    assert record["density_shortfall"] is True
    assert record["density_shortfall_vectors"] == 64 * 4096 - 32


def _repeat_digest(device: str) -> tuple[list[str], list[str]]:
    qname, weight, col_weights = _small_inputs(device)
    cw = col_weights[qname]
    first = bundle.learn_pool_v2(
        weight.unsqueeze(0), cw.unsqueeze(0), 4, qname=qname
    )
    second = bundle.learn_pool_v2(
        weight.unsqueeze(0), cw.unsqueeze(0), 4, qname=qname
    )
    return (
        [bundle.codebook_table_sha256(table) for table in first.tables],
        [bundle.codebook_table_sha256(table) for table in second.tables],
    )


def test_learned_v2_repeat_digest_is_exact_on_cpu():
    first, second = _repeat_digest("cpu")
    assert first == second


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_learned_v2_repeat_digest_is_exact_on_one_cuda_device():
    first, second = _repeat_digest("cuda")
    assert first == second


def test_v2_defaults_to_lattice_without_receipt(tmp_path, monkeypatch):
    qname, weight, col_weights = _small_inputs()

    def no_training(*_args, **_kwargs):
        raise AssertionError("unpromoted learned-v2 rung must not train")

    monkeypatch.setattr(bundle, "learn_pool_v2", no_training)
    loaded = bundle.train_and_save_bundle(
        tmp_path / "v2-lattice.pqcb",
        weights={qname: weight},
        col_weights=col_weights,
        formats=("FP8_CB_K4",),
        trainer_version="v2",
    )
    assert loaded.manifest["trainer"] == bundle.CB_LEARNED_TRAINER_V2_STAMP
    assert "promotion_receipt" not in loaded.manifest
    assert loaded.cell(qname, "FP8_CB_K4")["source"] == "lattice"


def test_v2_bundle_embeds_receipt_sampling_provenance_and_exact_tables(tmp_path):
    qname, weight, col_weights = _small_inputs()
    source_identity = _source_identity()
    receipt = _receipt(
        col_weights,
        learned_rungs={4},
        weights={qname: weight},
        source_identity=source_identity,
    )
    loaded = bundle.train_and_save_bundle(
        tmp_path / "v2-learned.pqcb",
        weights={qname: weight},
        col_weights=col_weights,
        formats=("FP8_CB_K4",),
        trainer_version="v2",
        promotion_receipt=receipt,
        **_promotion_kwargs(col_weights, source_identity),
    )
    cell = loaded.cell(qname, "FP8_CB_K4")
    assert cell["source"] == "learned"
    assert cell["rung_policy"]["status"] == "two_holdout_promoted"
    assert cell["training_provenance"]["schema"] == (
        bundle.CB_LEARNED_V2_SAMPLING_SCHEMA
    )
    assert cell["training_provenance"]["density_shortfall"] is False
    assert loaded.manifest["promotion_receipt"] == receipt
    assert bundle.load_bundle(loaded.path).bundle_content_sha256 == (
        loaded.bundle_content_sha256
    )

    with safe_open(str(loaded.path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    original_manifest = metadata[bundle.CB_LEARNED_BUNDLE_METADATA_KEY]
    manifest = json.loads(original_manifest)
    manifest["promotion_receipt"]["rungs"]["4"]["holdouts"][
        "holdout-b"
    ]["geomean_ratio"] = 1.0
    metadata[bundle.CB_LEARNED_BUNDLE_METADATA_KEY] = bundle._canonical_json(
        manifest
    )
    tampered = tmp_path / "v2-tampered-receipt.pqcb"
    save_file(tensors, str(tampered), metadata=metadata)
    with pytest.raises(ValueError, match="metrics require lattice"):
        bundle.load_bundle(tampered)

    manifest = json.loads(original_manifest)
    manifest["cells"][qname]["FP8_CB_K4"]["training_provenance"][
        "density_shortfall_vectors"
    ] = 1
    metadata[bundle.CB_LEARNED_BUNDLE_METADATA_KEY] = bundle._canonical_json(
        manifest
    )
    tampered = tmp_path / "v2-tampered-density.pqcb"
    save_file(tensors, str(tampered), metadata=metadata)
    with pytest.raises(ValueError, match="sampling arithmetic differs"):
        bundle.load_bundle(tampered)

    manifest = json.loads(original_manifest)
    manifest["promotion_receipt"]["candidate_codebooks"][qname]["4"][
        "subtable_content_sha256"
    ][0] = "f" * 64
    tampered_receipt = bundle.validate_promotion_receipt(
        manifest["promotion_receipt"],
        expected_model_id=source_identity["source"],
        expected_model_content_sha256=source_identity["content_sha256"],
        expected_calibration_hash="training-calibration",
        expected_imatrix_sha256=canonical_imatrix_sha256(col_weights),
        expected_role_census=role_census_for_qnames((qname,)),
        expected_qnames=(qname,),
    )
    manifest["rung_policy"] = {
        str(rung): policy
        for rung, policy in bundle.receipt_rung_policy(
            tampered_receipt
        ).items()
    }
    manifest["cells"][qname]["FP8_CB_K4"]["rung_policy"] = (
        manifest["rung_policy"]["4"]
    )
    metadata[bundle.CB_LEARNED_BUNDLE_METADATA_KEY] = bundle._canonical_json(
        manifest
    )
    tampered = tmp_path / "v2-tampered-candidate.pqcb"
    save_file(tensors, str(tampered), metadata=metadata)
    with pytest.raises(ValueError, match="exact promotion candidate"):
        bundle.load_bundle(tampered)


def test_v2_refuses_manual_promotion_or_wrong_imatrix_identity(tmp_path):
    qname, weight, col_weights = _small_inputs()
    with pytest.raises(ValueError, match="legacy off-law"):
        bundle.train_and_save_bundle(
            tmp_path / "off-law.pqcb",
            weights={qname: weight},
            col_weights=col_weights,
            formats=("FP8_CB_K29",),
            trainer_version="v2",
        )

    with pytest.raises(ValueError, match="receipt-derived source set"):
        bundle.train_and_save_bundle(
            tmp_path / "manual.pqcb",
            weights={qname: weight},
            col_weights=col_weights,
            formats=("FP8_CB_K4",),
            learned_formats=("FP8_CB_K4",),
            trainer_version="v2",
        )

    source_identity = _source_identity()
    shortfall_receipt = _receipt(
        col_weights,
        learned_rungs={48},
        weights={qname: weight},
        source_identity=source_identity,
    )
    with pytest.raises(ValueError, match="density shortfall.*lattice wins"):
        bundle.train_and_save_bundle(
            tmp_path / "shortfall.pqcb",
            weights={qname: weight},
            col_weights=col_weights,
            formats=("FP8_CB_K48",),
            trainer_version="v2",
            promotion_receipt=shortfall_receipt,
            **_promotion_kwargs(col_weights, source_identity),
        )
    assert not (tmp_path / "shortfall.pqcb").exists()

    receipt = _receipt(
        col_weights,
        learned_rungs={4},
        weights={qname: weight},
        source_identity=source_identity,
    )
    receipt["imatrix"]["value_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="value identity differs"):
        bundle.train_and_save_bundle(
            tmp_path / "wrong-imatrix.pqcb",
            weights={qname: weight},
            col_weights=col_weights,
            formats=("FP8_CB_K4",),
            trainer_version="v2",
            promotion_receipt=receipt,
            **_promotion_kwargs(col_weights, source_identity),
        )


def test_v2_receipt_refuses_missing_or_mismatched_source_probe_bindings(tmp_path):
    qname, weight, col_weights = _small_inputs()
    source_identity = _source_identity()
    receipt = _receipt(
        col_weights,
        learned_rungs={4},
        weights={qname: weight},
        source_identity=source_identity,
    )
    common = {
        "weights": {qname: weight},
        "col_weights": col_weights,
        "formats": ("FP8_CB_K4",),
        "trainer_version": "v2",
        "promotion_receipt": receipt,
    }
    with pytest.raises(ValueError, match="full streamed model identity"):
        bundle.train_and_save_bundle(tmp_path / "missing.pqcb", **common)

    other_identity = _source_identity(source="fixture/other-model")
    with pytest.raises(ValueError, match="model binding differs"):
        bundle.train_and_save_bundle(
            tmp_path / "wrong-model.pqcb",
            **common,
            **_promotion_kwargs(col_weights, other_identity),
        )

    bindings = _promotion_kwargs(col_weights, source_identity)
    bindings["probe_calibration_hash"] = "other-calibration"
    with pytest.raises(ValueError, match="calibration identity differs"):
        bundle.train_and_save_bundle(
            tmp_path / "wrong-probe.pqcb",
            **common,
            **bindings,
        )


def test_v2_rejects_arbitrary_supplied_tables_not_measured_by_receipt(tmp_path):
    qname, weight, col_weights = _small_inputs()
    source_identity = _source_identity()
    receipt = _receipt(
        col_weights,
        learned_rungs={4},
        weights={qname: weight},
        source_identity=source_identity,
    )
    arbitrary = tuple(torch.zeros(2, 2) for _ in range(4))
    output = tmp_path / "arbitrary-pretrained.pqcb"
    with pytest.raises(ValueError, match="exact promotion candidate"):
        bundle.train_and_save_bundle(
            output,
            weights={qname: weight},
            col_weights=col_weights,
            formats=("FP8_CB_K4",),
            trainer_version="v2",
            promotion_receipt=receipt,
            pretrained_codebooks={(qname, "FP8_CB_K4"): arbitrary},
            **_promotion_kwargs(col_weights, source_identity),
        )
    assert not output.exists()


def test_builder_gpu_guard_receives_selected_device(tmp_path, monkeypatch):
    observed: list[str] = []

    def guard(_component, device):
        observed.append(str(device))
        raise RuntimeError("guard sentinel")

    monkeypatch.setattr(
        "prismaquant.gpu_guard.require_cuda_hot_path",
        guard,
    )
    with pytest.raises(RuntimeError, match="guard sentinel"):
        builder.main([
            "--model-dir",
            str(tmp_path / "model"),
            "--col-weights",
            str(tmp_path / "col.pkl"),
            "--formats",
            "FP8_CB_K4",
            "--output",
            str(tmp_path / "out.pqcb"),
            "--device",
            "cpu",
        ])
    assert observed == ["cpu"]
