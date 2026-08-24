from __future__ import annotations

import json

import pytest

from prismaquant.rtx4090_fp8_burn import (
    BF16_FORMAT,
    CB_FORMATS,
    FULL_FORMATS,
    MEASURED_FORMATS,
    NATIVE_FP8_FORMAT,
    RENDER_FORMATS,
    RTX4090FP8BurnError,
    build_campaign_plan,
    validate_campaign_plan,
)


class _Profile:
    def fused_sibling_group(self, _qname):
        return None

    def packed_expert_format_group(self, _qname):
        return None


def _stats(*, reverse: bool = False):
    rows = []
    for layer in range(64):
        rows.extend((
            (
                f"model.layers.{layer}.self_attn.q_proj",
                {"n_params": 32, "in_features": 4, "out_features": 8},
            ),
            (
                f"model.layers.{layer}.mlp.down_proj",
                {"n_params": 32, "in_features": 8, "out_features": 4},
            ),
        ))
    if reverse:
        rows.reverse()
    return dict(rows)


def _plan(*, reverse: bool = False):
    return build_campaign_plan(
        _stats(reverse=reverse), profile=_Profile(),
        fixed_bf16={
            "lm_head": {
                "reason": "profile_pinned", "source_dtype": "bf16",
                "n_params": 100,
            },
            "mtp.proj": {
                "reason": "mtp_fixed", "source_dtype": "bf16",
                "n_params": 20,
            },
        },
        calibration={
            "calib_hash": "c" * 64, "nsamples": 32,
            "seqlen": 1024, "seed": 42,
        },
        bindings={
            "probe": {"sha256": "1" * 64},
            "col_weights": {"sha256": "2" * 64},
            "source_model_identity": {
                "sha256": "3" * 64, "content_sha256": "4" * 64,
            },
            "producer_snapshot": {"sha256": "5" * 64},
            "common_execution_attestation": {"sha256": "6" * 64},
            "dataset": {"sha256": "7" * 64},
        },
        source_dtype_census_sha256="8" * 64,
    )


def test_plan_has_exact_full_menu_and_terminal_contract():
    plan = _plan()
    assert tuple(plan["policy"]["formats"]) == FULL_FORMATS
    assert tuple(plan["policy"]["measured_formats"]) == MEASURED_FORMATS
    assert tuple(plan["policy"]["codebook_formats"]) == CB_FORMATS
    assert FULL_FORMATS[-2:] == (NATIVE_FP8_FORMAT, BF16_FORMAT)
    assert BF16_FORMAT not in MEASURED_FORMATS
    assert MEASURED_FORMATS == (
        "FP8_CB_K4",
        "FP8_CB_K16",
        "FP8_CB_K48",
        NATIVE_FP8_FORMAT,
    )

    for qname in plan["body"]["qnames"]:
        assert tuple(plan["maps"]["formats_by_qname"][qname]) == RENDER_FORMATS
        purposes = plan["maps"]["purposes_by_qname"][qname]
        assert set(purposes) == set(MEASURED_FORMATS)
        assert purposes == {
            "FP8_CB_K4": ["panel"],
            "FP8_CB_K16": ["anchor", "panel"],
            "FP8_CB_K48": ["panel"],
            NATIVE_FP8_FORMAT: ["anchor"],
        }
        assert plan["maps"]["unmeasured_formats_by_qname"][qname] == [
            BF16_FORMAT
        ]
        assert tuple(
            plan["maps"]["legal_cb_formats_by_qname"][qname]
        ) == CB_FORMATS


def test_plan_contains_no_out_of_lane_family_string():
    encoded = json.dumps(_plan(), sort_keys=True)
    disallowed = "NV" + "FP4"
    assert disallowed not in encoded


def test_plan_is_deterministic_across_probe_mapping_order():
    forward = _plan()
    reverse = _plan(reverse=True)
    assert forward == reverse
    assert forward["plan_sha256"] == reverse["plan_sha256"]


def test_contiguous_equal_lpt_tie_is_exact_disjoint_cover():
    plan = _plan()
    stripes = plan["stripes"]
    assert [row["layer_range"] for row in stripes] == [[0, 31], [32, 63]]
    assert plan["stripe_balance_proof"] == {
        **plan["stripe_balance_proof"],
        "strategy": "whole_layer_lpt_contiguous_equal_tie",
        "selected_metrics_exactly_equal": True,
        "selected_matches_lpt_loads": True,
    }
    first = set(stripes[0]["qnames"])
    second = set(stripes[1]["qnames"])
    assert first.isdisjoint(second)
    assert first | second == set(plan["body"]["qnames"])
    assert all(".layers." in name for name in first | second)
    assert all(int(name.split(".layers.", 1)[1].split(".", 1)[0]) < 32
               for name in first)
    assert all(int(name.split(".layers.", 1)[1].split(".", 1)[0]) >= 32
               for name in second)
    for metric in ("qnames", "parameters", "estimated_work", "render_cells"):
        assert (
            plan["stripe_balance_proof"]["selected"][0][metric]
            == plan["stripe_balance_proof"]["selected"][1][metric]
        )


def test_plan_validation_rejects_any_menu_edit():
    plan = _plan()
    plan["maps"]["formats_by_qname"][plan["body"]["qnames"][0]].pop()
    without_digest = dict(plan)
    without_digest.pop("plan_sha256")
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    plan["plan_sha256"] = canonical_json_sha256(
        without_digest, where="tampered test plan"
    )
    with pytest.raises(RTX4090FP8BurnError, match="maps"):
        validate_campaign_plan(plan)
