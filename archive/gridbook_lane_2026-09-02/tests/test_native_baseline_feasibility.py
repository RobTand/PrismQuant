from __future__ import annotations

import copy

import pytest

from prismaquant.native_baseline_feasibility import (
    SCHEMA,
    NativeConstructionUnit,
    build_native_baseline_certificate,
    canonical_sha256,
    certificate_sha256,
    validate_native_baseline_certificate,
    evaluate_expected_probe_census,
    _release_probe_census_contract,
)
from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile


class _SyntheticProfile:
    name = "synthetic"

    def fused_sibling_group(self, _qname: str):
        return None

    def packed_expert_format_group(self, qname: str):
        return "layer0-routed-experts" if ".experts." in qname else None


_GATE = "model.layers.0.mlp.experts.0.gate_proj"
_UP = "model.layers.0.mlp.experts.0.up_proj"
_DENSE = "model.layers.0.self_attn.q_proj"


def _inputs():
    # (8, 256) MXFP4_SOURCE is 1,088 exact bytes.  The dense block-FP8
    # source is 32,770 bytes; its cheapest legal native re-encode is NVFP4 at
    # 18,432 payload bytes + two exact fp32 global-scale scalars.
    stats = {
        _GATE: {"out_features": 8, "in_features": 256},
        _UP: {"out_features": 8, "in_features": 256},
        _DENSE: {"out_features": 128, "in_features": 256},
    }
    source_bytes = {_GATE: 1088, _UP: 1088, _DENSE: 32770}
    construction = NativeConstructionUnit(
        name="model.layers.1.ffn.experts",
        format_name="MXFP4_SOURCE",
        payload_bytes=200,
        physical_targets=("mtp.0.ffn.experts.0.w1",),
        source_span_ids=("mtp.0.ffn.experts.0.w1",),
        physical_target_payload_bytes=(200,),
    )
    census_counts = {
        "body_member_count": 3,
        "routed_member_count": 2,
        "nonexpert_member_count": 1,
        "routed_role_counts": {"gate_proj": 1, "up_proj": 1},
        "nonexpert_role_counts": {"q_proj": 1},
    }
    profile_binding = {
        "id": "synthetic",
        "structure_spec_sha256": "c" * 64,
        "implementation_sha256": "d" * 64,
    }
    profile_binding["identity_sha256"] = canonical_sha256(profile_binding)
    member_names_sha256 = canonical_sha256({"members": sorted(stats)})
    return dict(
        stats=stats,
        source_kinds={_GATE: "mxfp4", _UP: "mxfp4", _DENSE: "fp8_ue8m0"},
        source_bytes=source_bytes,
        source_span_ids={_GATE: ("gate",), _UP: ("up",), _DENSE: ("dense",)},
        model_profile=_SyntheticProfile(),
        target_profile="nvfp4_cb",
        lane_id="nvfp4_cb",
        budget_bytes=21000,
        source_checkpoint_payload_bytes=sum(source_bytes.values()) + 200 + 1000,
        source_model_binding={
            "identity_schema": "prismaquant.streamed_model.identity.v1",
            "content_sha256": "a" * 64,
            "identity_sha256": "b" * 64,
            "shard_count": 1,
        },
        contract_binding={
            "model_profile": profile_binding,
            "serving_profile": {"id": "nvfp4_cb", "spec_sha256": "e" * 64},
            "lane": {
                "id": "nvfp4_cb",
                "export_container": "nvfp4_cb",
                "spec_sha256": "f" * 64,
            },
            "gridbook_runtime_pin": {
                "schema": "prismaquant.gridbook_runtime_pin.v3",
                "repository": "https://example.invalid/gridbook",
                "commit": "1" * 40,
                "version": "1.2.3",
                "version_is_release": True,
                "runtime_contract_schema": "gridbook.runtime-contract.v3",
                "required_abi_features": {
                    "routed_moe_per_role_codebook_lut": 1,
                    "source_fp8_block128_w8a16": 1,
                },
                "file_sha256": "9" * 64,
            },
        },
        probe_census={
            "schema": "prismaquant.expected_probe_census.v1",
            "contract_id": "synthetic.census.v1",
            "authority_sha256": "8" * 64,
            "profile_id": "synthetic",
            "classifier": "profile_declared_routed_expert.v1",
            "routed_qname_regex": (
                r"^model[.]layers[.][0-9]+[.]mlp[.]experts[.][0-9]+[.]"
                r"(?:gate|up|down)_proj$"
            ),
            "expected_member_names_sha256": member_names_sha256,
            "observed_member_names_sha256": member_names_sha256,
            "expected": census_counts,
            "observed": copy.deepcopy(census_counts),
            "complete": True,
        },
        construction_units=(construction,),
    )


def test_exact_lower_bound_groups_serving_units_and_proves_excess():
    certificate = build_native_baseline_certificate(**_inputs())

    assert certificate["schema"] == SCHEMA
    assert certificate["status"] == "infeasible"
    assert certificate["coverage"] == {
        "complete": True,
        "body_member_count": 3,
        "body_serving_unit_count": 2,
        "construction_unit_count": 1,
        "body_members_sha256": certificate["coverage"]["body_members_sha256"],
        "construction_members_sha256": certificate["coverage"]["construction_members_sha256"],
        "source_spans_are_disjoint": True,
        "probe_census": certificate["coverage"]["probe_census"],
    }
    assert certificate["accounting"] == {
        "source_checkpoint_payload_bytes": 36146,
        "body_source_payload_bytes": 34946,
        "construction_source_payload_bytes": 200,
        "immutable_floor_bytes": 1000,
        "body_no_cb_lower_bound_bytes": 20616,
        "construction_no_cb_lower_bound_bytes": 200,
        "all_native_lower_bound_bytes": 21816,
        "budget_bytes": 21000,
        "excess_bytes": 816,
    }
    routed = next(row for row in certificate["units"] if row["kind"] == "serving" and row["packed_expert"])
    assert routed["members"] == [_GATE, _UP]
    assert routed["available_no_cb_formats"] == [
        {"format": "MXFP4_SOURCE", "payload_bytes": 2176}
    ]
    dense = next(row for row in certificate["units"] if row["name"] == _DENSE)
    assert dense["lower_bound_format"] == "NVFP4"
    assert dense["lower_bound_payload_bytes"] == 18440
    validate_native_baseline_certificate(certificate)


def test_rejects_incomplete_body_coverage():
    inputs = _inputs()
    del inputs["source_kinds"][_UP]
    with pytest.raises(ValueError, match="source_kinds coverage differs"):
        build_native_baseline_certificate(**inputs)


def test_rejects_overlapping_source_spans_across_body_and_construction():
    inputs = _inputs()
    old = inputs["construction_units"][0]
    inputs["construction_units"] = (
        NativeConstructionUnit(
            old.name,
            old.format_name,
            old.payload_bytes,
            old.physical_targets,
            ("gate", "mtp-scale"),
            old.physical_target_payload_bytes,
        ),
    )
    with pytest.raises(ValueError, match="charged by both"):
        build_native_baseline_certificate(**inputs)


def test_refuses_a_non_infeasible_budget_and_detects_digest_tampering():
    inputs = _inputs()
    inputs["budget_bytes"] = 21816
    with pytest.raises(ValueError, match="infeasibility is not proven"):
        build_native_baseline_certificate(**inputs)

    certificate = build_native_baseline_certificate(**_inputs())
    assert certificate["certificate_sha256"] == certificate_sha256(certificate)
    tampered = copy.deepcopy(certificate)
    tampered["accounting"]["excess_bytes"] += 1
    with pytest.raises(ValueError, match="does not match canonical"):
        validate_native_baseline_certificate(tampered)


def test_rejects_fabricated_lower_bound_even_with_recomputed_digest():
    certificate = build_native_baseline_certificate(**_inputs())
    fabricated = copy.deepcopy(certificate)
    fabricated["accounting"]["all_native_lower_bound_bytes"] += 100
    fabricated["accounting"]["excess_bytes"] += 100
    fabricated["proof"]["excess_bytes"] += 100
    fabricated["certificate_sha256"] = certificate_sha256(fabricated)

    with pytest.raises(ValueError, match="reconstructed from the unit ledger"):
        validate_native_baseline_certificate(fabricated)


def test_rejects_fabricated_unit_minimum_and_consistent_totals():
    certificate = build_native_baseline_certificate(**_inputs())
    fabricated = copy.deepcopy(certificate)
    dense = next(row for row in fabricated["units"] if row["name"] == _DENSE)
    dense["available_no_cb_formats"][0]["payload_bytes"] -= 100
    dense["lower_bound_payload_bytes"] -= 100
    fabricated["accounting"]["body_no_cb_lower_bound_bytes"] -= 100
    fabricated["accounting"]["all_native_lower_bound_bytes"] -= 100
    fabricated["accounting"]["excess_bytes"] -= 100
    fabricated["proof"]["excess_bytes"] -= 100
    fabricated["certificate_sha256"] = certificate_sha256(fabricated)

    with pytest.raises(ValueError, match="exact legality/bytes"):
        validate_native_baseline_certificate(fabricated)


def test_expected_probe_census_rejects_dropped_dense_member():
    inputs = _inputs()
    del inputs["stats"][_DENSE]
    del inputs["source_kinds"][_DENSE]
    del inputs["source_bytes"][_DENSE]
    del inputs["source_span_ids"][_DENSE]

    with pytest.raises(ValueError, match="probe_census expected/observed"):
        build_native_baseline_certificate(**inputs)


def test_real_dsv4_census_contract_rejects_one_dropped_dense_member():
    qnames = []
    for layer in range(43):
        for expert in range(256):
            for role in ("gate_proj", "up_proj", "down_proj"):
                qnames.append(
                    f"model.layers.{layer}.mlp.experts.{expert}.{role}"
                )
        for role in ("gate_proj", "up_proj", "down_proj"):
            qnames.append(f"model.layers.{layer}.mlp.shared_experts.{role}")
        for role in ("wq_a", "wq_b", "wkv", "wo_b"):
            qnames.append(f"model.layers.{layer}.self_attn.{role}")

    profile = DeepseekV4Profile()
    contract = _release_probe_census_contract(profile)
    census = evaluate_expected_probe_census(qnames, profile, contract)
    assert census["observed"] == {
        "body_member_count": 33_325,
        "routed_member_count": 33_024,
        "nonexpert_member_count": 301,
        "routed_role_counts": {
            "down_proj": 11_008,
            "gate_proj": 11_008,
            "up_proj": 11_008,
        },
        "nonexpert_role_counts": {
            "down_proj": 43,
            "gate_proj": 43,
            "up_proj": 43,
            "wkv": 43,
            "wo_b": 43,
            "wq_a": 43,
            "wq_b": 43,
        },
    }

    dropped = [name for name in qnames if name != "model.layers.42.self_attn.wo_b"]
    with pytest.raises(ValueError, match="probe census is incomplete"):
        evaluate_expected_probe_census(dropped, profile, contract)
