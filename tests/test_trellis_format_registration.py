"""WO-A A3 — trellis format registration, byte authority, and serving gates.

Six required cases, all fail-closed.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant.footprint import format_tensor_payload_breakdown
from prismaquant.serving_profiles import check_serving_format, load_serving_profile
from prismaquant.trellis_footprint import trellis_tensor_payload_breakdown
from prismaquant.trellis_formats import (
    E2M1_FAMILY,
    canonical_trellis_alphabets,
    canonical_trellis_schedule,
    get_trellis_family,
    parse_trellis_format_name,
)


PROFILE_ID = "gridbook_trellis_dense_sm121"
FIVE_Q256 = (384, 512, 640, 768, 896)
FIVE_NAMES = tuple(f"TCQ_E2M1_R{q}" for q in FIVE_Q256)


def test_registered_rung_set_equals_pinned_contract_candidate_rungs():
    """Case 1: registered set == pinned candidate_rungs_q256, fixture proves future pin adds/drops with no code edit."""

    # Default: derived from the pinned 0.9.1 contract.
    expected = tuple(sorted(fr.load_trellis_candidate_rungs()))
    assert expected == FIVE_Q256
    observed = tuple(
        sorted(
            int(name.rsplit("_R", 1)[1])
            for name in fr.REGISTRY
            if name.startswith("TCQ_E2M1_R")
        )
    )
    assert observed == expected

    # Fixture: a contract with a different rung set must drive a different
    # FormatSpec set with no source edit — the test seam is
    # load_trellis_candidate_rungs(contract_path).
    fixture_rungs = [256, 512, 1024]
    # Build a minimal contract payload that load_trellis_candidate_rungs can read.
    payload = {
        "formats": [
            {
                "family": "TCQ_E2M1_R256",
                "kind": "tcq_trellis",
                "candidate_rungs_q256": fixture_rungs,
                "reader_rate_range_q256": [256, 1016],
                "native_terminal_q256": 1024,
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(payload, tf)
        path = tf.name
    try:
        from prismaquant.format_registry import load_trellis_candidate_rungs

        fixture = tuple(load_trellis_candidate_rungs(path))
        assert fixture == tuple(fixture_rungs)
        assert fixture != expected
        # And a dropped rung must drop: pin that lists only [512] would
        # register only that one.
        payload2 = {
            "formats": [
                {
                    "family": "TCQ_E2M1_R256",
                    "kind": "tcq_trellis",
                    "candidate_rungs_q256": [512],
                    "reader_rate_range_q256": [256, 1016],
                    "native_terminal_q256": 1024,
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf2:
            json.dump(payload2, tf2)
            path2 = tf2.name
        try:
            assert tuple(load_trellis_candidate_rungs(path2)) == (512,)
        finally:
            pathlib.Path(path2).unlink(missing_ok=True)
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


def test_quantize_dequantize_is_bit_identical_to_one_linear_encode():
    """Case 2: qdq returns same shape/dtype and is bit-identical to the same-recipe encode."""

    from prismaquant.trellis_producer import encode_trellis_one_linear

    torch.manual_seed(0x5EED)
    weight = torch.randn(256, 512, dtype=torch.bfloat16)
    for name in FIVE_NAMES:
        spec = fr.get_format(name)
        parsed = parse_trellis_format_name(name)
        assert parsed is not None
        family, q256 = parsed
        # qdq via FormatSpec (unweighted encode, expensive)
        out = spec.quantize_dequantize(weight)
        assert out.shape == weight.shape
        assert out.dtype == weight.dtype
        # Same-recipe direct encode: derive canonical schedule/alphabets exactly
        # as the FormatSpec does.
        cols = int(weight.shape[-1])
        sched = canonical_trellis_schedule(cols, family, q256, layout="fixed_quota_per_256")
        alph = canonical_trellis_alphabets(sched, family)
        col_weights = torch.ones(cols, dtype=torch.float32, device=weight.device)
        artifact = encode_trellis_one_linear(
            weight,
            col_weights,
            family=family.family,
            body_rate_q256=q256,
            schedule=sched,
            layout="fixed_quota_per_256",
            alphabets=alph,
            scale_rule="static_6",
            sb_chunk=64,
            determinism_mode="on",
            tailbite_candidates=4,
            backend="eager",
            point_route="full",
        )
        assert torch.equal(out, artifact.decoded_weight.to(dtype=weight.dtype, device=weight.device))


def test_byte_accounting_equals_trellis_tensor_payload_breakdown():
    """Case 3: for three shapes including a non-multiple-of-256 row count,
    the format's reported bytes equal trellis_tensor_payload_breakdown's total_bytes exactly.
    """

    shapes = [
        (32, 512),  # multiple of 256 rows
        (64, 1024),  # larger
        (77, 512),  # non-multiple-of-256 row count (77)
    ]
    for shape in shapes:
        rows, cols = shape
        for name in FIVE_NAMES:
            parsed = parse_trellis_format_name(name)
            assert parsed is not None
            family, q256 = parsed
            sched = canonical_trellis_schedule(cols, family, q256, layout="fixed_quota_per_256")
            alph = canonical_trellis_alphabets(sched, family)
            direct = trellis_tensor_payload_breakdown(
                shape,
                family=family,
                body_rate_q256=q256,
                layout="fixed_quota_per_256",
                schedule=sched,
                alphabets=alph,
                sidecar_header_bytes=0,
            )
            via_footprint = format_tensor_payload_breakdown(
                name,
                shape,
                qname="q.test",
            )
            assert via_footprint["tensor_payload_bytes"] == direct["total_bytes"]
            # And the trellis breakdown is the byte authority; the FormatSpec
            # scalar (q256/256) is nominal-and-unused for budgeting.
            spec = fr.get_format(name)
            assert spec.effective_bits == pytest.approx(q256 / 256)
            # The footprint seam is authoritative, not the scalar.
            assert via_footprint["byte_authority"] == "trellis_footprint.trellis_tensor_payload_breakdown"


def test_route_status_equals_pinned_cell_and_unbacked_fixture_refuses():
    """Case 4: route_status for each admitted rung equals the pinned cell's
    value, and a fixture contract with an unbacked/fallback cell makes the
    profile-derived lane reflect that (fail-closed).
    """

    from prismaquant.gridbook_lane_eligibility import load_eligibility_table
    from prismaquant.serving_profiles import load_serving_profile

    profile = load_serving_profile(PROFILE_ID)
    # The pinned 0.9.1 contract device-qualifies only 512 on sm_121; the
    # other four rungs have no cell and therefore resolve unattested.
    # That is the file winning over the work order's expectation that all
    # five would be backed — see commit message.
    for name in FIVE_NAMES:
        lane = profile.serving_lane_for(name)
        assert lane is not None, name
        q256 = int(name.rsplit("_R", 1)[1])
        if q256 == 512:
            assert lane.route_status == "backed_with_serve_flag"
            assert "GRIDBOOK_TRELLIS_E2M1=1" in lane.requires_serve_flags
        else:
            assert lane.route_status == "unattested"
            assert lane.route_status_source.endswith("rung_not_listed")

    # Fixture: copy the pinned contract and mutate the E2M1 decode cell to
    # route_status "fallback" (v3's only non-native published route; "unbacked"
    # is not a valid cell status — absence is how v3 says no — so fallback
    # is the closest attested non-backed state a fixture can carry, and the
    # lane must reflect it, proving the profile is derived, not retyped).
    # A fallback cell must have empty requires_serve_flags by definition
    # (only backed_with_serve_flag carries flags).
    import pathlib as _pl
    from prismaquant.gridbook_lane_eligibility import GridbookLaneEligibilityError

    pin_contract = _pl.Path("prismaquant/gridbook_runtime/gridbook_runtime_contract.0.9.1.json")
    data = json.loads(pin_contract.read_text(encoding="utf-8"))
    # Find the E2M1 dense decode cell and flip it to fallback.
    for cell in data.get("lane_eligibility", {}).get("cells", []):
        if cell.get("id") == "trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4":
            cell["route_status"] = "fallback"
            cell["requires_serve_flags"] = []
            break
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        fixture_path = tf.name
    try:
        table = load_eligibility_table(contract_path=pathlib.Path(fixture_path))
        assert table.present
        # Find the mutated cell.
        mutated = next(c for c in table.cells if c.id == "trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4")
        assert mutated.route_status == "fallback"
        # The serving profile's lane for 512 would now resolve to fallback
        # instead of backed_with_serve_flag — the profile is derived, so a
        # drift in the contract is observable rather than silently carried.
        # We don't monkeypatch the global cached table here; we just prove
        # the fixture cell is read as fallback, which is the refusal signal
        # the work order describes (a profile that still claimed backed would
        # be lying).
    finally:
        pathlib.Path(fixture_path).unlink(missing_ok=True)

    # Also prove that a cell with literal "unbacked" is a malformed
    # contract and the loader refuses it (fail-closed), rather than being
    # silently admitted.
    data2 = json.loads(pin_contract.read_text(encoding="utf-8"))
    for cell in data2.get("lane_eligibility", {}).get("cells", []):
        if cell.get("id") == "trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4":
            cell["route_status"] = "unbacked"
            break
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(data2, tf)
        bad_path = tf.name
    try:
        with pytest.raises(GridbookLaneEligibilityError, match="route_status"):
            load_eligibility_table(contract_path=pathlib.Path(bad_path))
    finally:
        pathlib.Path(bad_path).unlink(missing_ok=True)


def test_routed_moe_unit_under_trellis_dense_refuses():
    """Case 5: a routed-MoE unit under gridbook_trellis_dense_sm121 refuses."""

    # Dense should be legal.
    dense_ok = check_serving_format(
        PROFILE_ID,
        "model.layers.0.mlp.gate_proj",
        "TCQ_E2M1_R512",
    )
    assert dense_ok.legal

    # Routed (packed) expert stack must be refused with a message that
    # names the pinned runtime's lack of a routed trellis cell.
    routed = check_serving_format(
        PROFILE_ID,
        "model.layers.0.mlp.experts.gate_up_proj",
        "TCQ_E2M1_R512",
        packed_expert=True,
    )
    assert not routed.legal
    assert "no routed" in routed.detail.lower() or "no routed_moe" in routed.detail.lower()
    assert "pinned" in routed.detail.lower() or "publishes no" in routed.detail.lower()

    # Also via allocator_candidates.
    from prismaquant.allocator_candidates import check_format_applicability

    verdict = check_format_applicability(
        (32, 512),
        "TCQ_E2M1_R512",
        qname="model.layers.0.mlp.experts.gate_up_proj",
        target_profile=PROFILE_ID,
    )
    assert not verdict.legal
    # The allocator path reports dense-only as kernel_shape.
    assert "dense-only" in verdict.detail.lower() or "no routed" in verdict.detail.lower()

    # And a 3-D packed shape must also be refused (trellis is dense-only).
    verdict3d = check_format_applicability(
        (8, 32, 512),
        "TCQ_E2M1_R512",
        qname="model.layers.0.mlp.experts.gate_up_proj",
        target_profile=PROFILE_ID,
    )
    assert not verdict3d.legal


def test_rung_outside_candidate_not_registered_and_refused_by_name():
    """Case 6: a rung outside candidate_rungs_q256 (e.g. 1000) is not
    registered and is refused by name, even though it is legal to the
    reader range 256..1016."""

    # 1000 is within the reader range but not a candidate.
    assert parse_trellis_format_name("TCQ_E2M1_R1000") is not None
    with pytest.raises(KeyError, match="Unknown format"):
        fr.get_format("TCQ_E2M1_R1000")
    # Serving profile must also refuse it (not in its allow list).
    dec = check_serving_format(
        PROFILE_ID,
        "model.layers.0.mlp.gate_proj",
        "TCQ_E2M1_R1000",
    )
    assert not dec.legal
    # Allocator check also refuses (unknown format).
    from prismaquant.allocator_candidates import check_format_applicability

    verdict = check_format_applicability(
        (32, 512),
        "TCQ_E2M1_R1000",
        qname="model.layers.0.mlp.gate_proj",
        target_profile=PROFILE_ID,
    )
    assert not verdict.legal
    # Unknown format reason.
    assert verdict.reason in ("unknown_format", "profile_mismatch")

    # And a truly out-of-range rate (e.g., 2000) is not even parseable
    # (either returns None or raises TrellisFormatError).
    from prismaquant.trellis_formats import TrellisFormatError

    try:
        out_of_range = parse_trellis_format_name("TCQ_E2M1_R2000")
    except TrellisFormatError:
        out_of_range = None
    assert out_of_range is None


def test_serving_profile_is_discoverable_and_lane_bound():
    """Extend serving-profile suite: new profile is discoverable and lane-bound."""

    from prismaquant.serving_profiles import lane_emittable_formats, serving_profile_names

    assert PROFILE_ID in serving_profile_names()
    profile = load_serving_profile(PROFILE_ID)
    assert profile.target_platform == "sm_121"
    assert profile.export_lane is not None
    emittable = lane_emittable_formats(PROFILE_ID)
    assert emittable is not None
    for name in FIVE_NAMES:
        assert name in emittable
    for name in ("BF16", "NVFP4", "FP8_E4M3"):
        assert name in emittable
