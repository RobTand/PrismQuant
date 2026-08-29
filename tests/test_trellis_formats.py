from __future__ import annotations

import pytest

from prismaquant.trellis_formats import (
    E2M1_FAMILY,
    E4M3_FAMILY,
    FAMILIES,
    LAYOUT_FIXED_QUOTA,
    LAYOUT_TIGHT_OFFSETS,
    RATE_SURFACE_ADAPTIVE,
    RATE_SURFACE_ALL_LEGAL,
    RATE_SURFACE_DENSE,
    TrellisFormatError,
    TrellisRateSurface,
    format_contract_payload,
    native_code_value,
    parse_trellis_format_name,
    quality_candidate_format_names,
    trellis_rate_surface,
    validate_alphabets,
    validate_body_rate_q256,
    validate_schedule,
)
from prismaquant.trellis_footprint import trellis_tensor_payload_breakdown


def _alphabet(rate: int, grid_bits: int) -> list[int]:
    return list(range(1 << (rate + 1)))


def test_contract_publishes_candidates_as_research_only():
    payload = format_contract_payload()
    assert payload["wire_schema"] == "gridbook.trellis.wire.v1"
    assert payload["generator_octal"] == ["561", "753"]
    rows = {row["family"]: row for row in payload["families"]}
    assert rows[E2M1_FAMILY]["quality_candidate_q256"] == [
        384, 512, 640, 768, 896,
    ]
    assert rows[E4M3_FAMILY]["quality_candidate_q256"] == [1152]
    assert rows[E2M1_FAMILY]["research_q256_bounds_inclusive"] == [256, 1016]
    assert rows[E4M3_FAMILY]["research_q256_bounds_inclusive"] == [256, 2040]
    assert all(row["producer_eligible"] is False for row in rows.values())
    assert payload["allocator_rate_surface"] == {
        "modes": ["all_legal", "dense"],
        "adaptive_mode": "adaptive",
        "address_fields": [
                "family",
                "body_rate_q256",
                "layout",
                "pre_render_recipe_identity_sha256",
        ],
        "public_format_registry_entries_created": 0,
        "producer_eligible": False,
    }
    assert set(quality_candidate_format_names()) == {
        "TCQ_E2M1_R384", "TCQ_E2M1_R512", "TCQ_E2M1_R640",
        "TCQ_E2M1_R768", "TCQ_E2M1_R896", "TCQ_E4M3_R1152",
    }


def test_all_legal_and_dense_rate_surfaces_are_explicit_and_deterministic():
    all_legal = trellis_rate_surface(
        E2M1_FAMILY,
        mode=RATE_SURFACE_ALL_LEGAL,
    )
    assert tuple(all_legal.rates_q256) == tuple(range(256, 1017))
    assert len(all_legal.rates_q256) == 761
    assert all_legal.rates_q256.__sizeof__() < 128
    assert "rates_q256" not in all_legal.as_dict()
    assert all_legal.as_dict()["rate_count"] == 761
    assert all_legal.as_dict()["public_format_registry_entries_created"] == 0
    assert all_legal.as_dict()["producer_eligible"] is False
    assert all_legal.identity_sha256 == trellis_rate_surface(
        E2M1_FAMILY,
        mode=RATE_SURFACE_ALL_LEGAL,
    ).identity_sha256
    fp8_all_legal = trellis_rate_surface(
        E4M3_FAMILY,
        mode=RATE_SURFACE_ALL_LEGAL,
    )
    assert len(fp8_all_legal.rates_q256) == 1785
    assert fp8_all_legal.rates_q256[0] == 256
    assert fp8_all_legal.rates_q256[-1] == 2040

    dense = trellis_rate_surface(
        E2M1_FAMILY,
        mode=RATE_SURFACE_DENSE,
        start_q256=256,
        stop_q256=265,
        step_q256=4,
        include_q256=(258,),
    )
    assert tuple(dense.rates_q256) == (256, 258, 260, 264, 265)
    assert dense.anchor_q256 == (258,)
    assert dense.identity_sha256 != all_legal.identity_sha256
    assert dense.rates_q256[::-1] == (265, 264, 260, 258, 256)
    assert dense.rates_q256[4:0:-2] == (265, 260)


def test_rate_surface_refuses_implicit_expansion_and_noninteger_anchors():
    with pytest.raises(TrellisFormatError, match="step_q256"):
        trellis_rate_surface(
            E2M1_FAMILY,
            mode=RATE_SURFACE_DENSE,
        )
    with pytest.raises(TrellisFormatError, match="include_q256 entry"):
        trellis_rate_surface(
            E2M1_FAMILY,
            mode=RATE_SURFACE_DENSE,
            step_q256=16,
            include_q256=(True,),
        )


def test_rate_surface_copies_every_mutable_sequence_before_identity():
    bounds = [256, 1016]
    anchors = [384, 640]
    proposals = [512]
    surface = TrellisRateSurface(
        family=E2M1_FAMILY,
        mode=RATE_SURFACE_ADAPTIVE,
        bounds_q256=bounds,
        anchor_q256=anchors,
        proposed_q256=proposals,
        source_identity_sha256="a" * 64,
    )
    identity = surface.identity_sha256
    bounds[0] = 999
    anchors[0] = 999
    proposals[0] = 999
    serialized = surface.as_dict()
    serialized["bounds_q256_inclusive"][0] = 999
    serialized["anchor_q256"][0] = 999
    assert surface.bounds_q256 == (256, 1016)
    assert surface.anchor_q256 == (384, 640)
    assert surface.proposed_q256 == (512,)
    assert surface.identity_sha256 == identity


@pytest.mark.parametrize(
    "name,family,rate",
    [
        ("TCQ_E2M1_R384", E2M1_FAMILY, 384),
        ("TCQ_E4M3_R1536", E4M3_FAMILY, 1536),
    ],
)
def test_format_name_round_trip(name, family, rate):
    parsed = parse_trellis_format_name(name)
    assert parsed == (FAMILIES[family], rate)
    assert parsed[0].format_name(parsed[1]) == name
    assert parse_trellis_format_name(name.lower()) is None


def test_mathematical_endpoints_exclude_native_scalar_terminals():
    assert FAMILIES[E2M1_FAMILY].mathematical_q256_bounds == (256, 1016)
    assert FAMILIES[E4M3_FAMILY].mathematical_q256_bounds == (256, 2040)
    validate_body_rate_q256(E2M1_FAMILY, 1016)
    validate_body_rate_q256(E4M3_FAMILY, 2040)
    with pytest.raises(TrellisFormatError, match=r"\[256, 1016\]"):
        validate_body_rate_q256(E2M1_FAMILY, 1024)
    with pytest.raises(TrellisFormatError, match=r"\[256, 2040\]"):
        validate_body_rate_q256(E4M3_FAMILY, 2048)


def test_fixed_quota_requires_exact_per_block_budget():
    schedule = [1, 3] * 256
    assert validate_schedule(
        E2M1_FAMILY, 512, schedule, layout=LAYOUT_FIXED_QUOTA,
    ) == tuple(schedule)
    schedule[0], schedule[257] = 2, 2
    # Tensor-wide rate is unchanged, but the first and second blocks differ.
    with pytest.raises(TrellisFormatError, match="fixed-quota block 0"):
        validate_schedule(
            E2M1_FAMILY, 512, schedule, layout=LAYOUT_FIXED_QUOTA,
        )
    assert validate_schedule(
        E2M1_FAMILY, 512, schedule, layout=LAYOUT_TIGHT_OFFSETS,
    ) == tuple(schedule)


def test_short_tailbiting_cycle_fails_at_seven_and_passes_at_eight():
    seven = [4] * 256
    seven[:7] = [3] * 7
    nine = [4] * 256
    nine[:9] = [3] * 9
    with pytest.raises(TrellisFormatError, match="has 7 coded steps"):
        validate_schedule(
            E2M1_FAMILY, 1016, [*seven, *nine], layout=LAYOUT_TIGHT_OFFSETS,
        )
    eight = [4] * 256
    eight[:8] = [3] * 8
    assert validate_schedule(
        E2M1_FAMILY, 1016, eight, layout=LAYOUT_FIXED_QUOTA,
    ) == tuple(eight)


def test_short_tail_matches_gridbook_layout_semantics():
    # Fixed quota binds complete 256-column blocks.  The short physical tail
    # remains a full schedule (and must itself contain >=8 coded steps), but it
    # is charged exactly rather than being padded to the nominal q256 quota.
    schedule = [2] * 256 + [1] * 8
    assert validate_schedule(
        E2M1_FAMILY, 512, schedule, layout=LAYOUT_FIXED_QUOTA,
    ) == tuple(schedule)
    with pytest.raises(TrellisFormatError, match="at least one physical body bit"):
        validate_schedule(
            E2M1_FAMILY, 512, schedule, layout=LAYOUT_TIGHT_OFFSETS,
        )

    result = trellis_tensor_payload_breakdown(
        (2, 264),
        family=E2M1_FAMILY,
        body_rate_q256=512,
        layout=LAYOUT_FIXED_QUOTA,
        schedule=schedule,
        alphabets={1: _alphabet(1, 4), 2: _alphabet(2, 4)},
    )
    assert result["block_count"] == 2
    assert result["body_bits_per_row"] == 520
    assert result["body_row_stride_bytes"] == 80
    assert result["scale_bytes"] == 34
    assert result["schedule_bytes"] == 132
    assert result["block_offset_bytes"] == 0


def test_short_tight_schedule_allows_sub_bit_q256_residual():
    # 527 physical bits versus 511/256 * 264 columns leaves a q256 residual
    # of 8, i.e. 1/32 of one physical body bit.
    schedule = [2] * 263 + [1]
    assert validate_schedule(
        E2M1_FAMILY, 511, schedule, layout=LAYOUT_TIGHT_OFFSETS,
    ) == tuple(schedule)


def test_e4m3_r7_allows_duplicates_but_refuses_nan_bytes():
    schedule = [7] * 256
    finite = [code for code in range(256) if code not in {0x7F, 0xFF}]
    codes = [*finite, 0x00, 0x80]
    codes.sort(key=lambda code: (native_code_value(E4M3_FAMILY, code), code))
    assert len(codes) == 256
    assert validate_alphabets(E4M3_FAMILY, schedule, {7: codes})[7] == tuple(codes)
    codes[-1] = 0x7F
    with pytest.raises(TrellisFormatError, match="cannot emit NaNs"):
        validate_alphabets(E4M3_FAMILY, schedule, {7: codes})


def test_e2m1_tight_offset_footprint_charges_every_plane():
    schedule = [2] * 256
    result = trellis_tensor_payload_breakdown(
        (8, 256),
        family=E2M1_FAMILY,
        body_rate_q256=512,
        layout=LAYOUT_TIGHT_OFFSETS,
        schedule=schedule,
        alphabets={2: _alphabet(2, 4)},
    )
    assert result["body_bpw"] == 2.0
    assert result["body_bytes"] == 512
    assert result["body_padding_bytes"] == 0
    assert result["scale_bytes"] == 128
    assert result["wire_header_bytes"] == 88
    assert result["schedule_bytes"] == 128
    assert result["block_offset_bytes"] == 8
    assert result["alphabet_bytes"] == 11
    assert result["structural_side_information_bytes"] == 235
    assert result["wire_side_information_bytes"] == 363
    assert result["side_information_bytes"] == 363
    assert result["total_bytes"] == 875
    assert result["exact_bpw"] == 3.41796875
    assert result["expanded_weight_resident_bytes"] == 0
    assert result["producer_eligible"] is False

    with_sidecar = trellis_tensor_payload_breakdown(
        (8, 256),
        family=E2M1_FAMILY,
        body_rate_q256=512,
        layout=LAYOUT_TIGHT_OFFSETS,
        schedule=schedule,
        alphabets={2: _alphabet(2, 4)},
        sidecar_header_bytes=7,
    )
    assert with_sidecar["wire_side_information_bytes"] == 363
    assert with_sidecar["side_information_bytes"] == 370
    assert with_sidecar["total_bytes"] == 882


def test_pre_render_recipe_identity_binds_known_layout_not_physical_wire():
    first = trellis_tensor_payload_breakdown(
        (8, 256),
        family=E2M1_FAMILY,
        body_rate_q256=512,
        layout=LAYOUT_TIGHT_OFFSETS,
        schedule=[1, 3] * 128,
        alphabets={1: _alphabet(1, 4), 3: [
            15, 14, 13, 12, 11, 10, 9, 0,
            8, 1, 2, 3, 4, 5, 6, 7,
        ]},
    )
    second = trellis_tensor_payload_breakdown(
        (8, 256),
        family=E2M1_FAMILY,
        body_rate_q256=512,
        layout=LAYOUT_TIGHT_OFFSETS,
        schedule=[3, 1] * 128,
        alphabets={1: _alphabet(1, 4), 3: [
            15, 14, 13, 12, 11, 10, 9, 0,
            8, 1, 2, 3, 4, 5, 6, 7,
        ]},
    )
    assert first["total_bytes"] == second["total_bytes"]
    assert first["schedule_identity_sha256"] != second[
        "schedule_identity_sha256"
    ]
    assert first["pre_render_recipe_identity_sha256"] != second[
        "pre_render_recipe_identity_sha256"
    ]
    assert first["rendered_wire_identity_sha256"] is None
    assert first["pre_render_recipe_identity_scope"] == (
        "layout_and_byte_recipe_without_encoded_body_or_scale_values"
    )


def test_fp8_fixed_quota_footprint_is_shape_and_side_inclusive():
    block = [3, 4] * 128
    schedule = block * 2
    result = trellis_tensor_payload_breakdown(
        (4, 512),
        family=E4M3_FAMILY,
        body_rate_q256=896,
        layout=LAYOUT_FIXED_QUOTA,
        schedule=schedule,
        alphabets={3: _alphabet(3, 8), 4: _alphabet(4, 8)},
    )
    assert result["body_bpw"] == 3.5
    assert result["body_row_stride_bytes"] == 224
    assert result["body_bytes"] == 896
    assert result["scale_bytes"] == 16
    assert result["wire_header_bytes"] == 88
    assert result["schedule_bytes"] == 256
    assert result["block_offset_bytes"] == 0
    assert result["alphabet_bytes"] == 54
    assert result["structural_side_information_bytes"] == 398
    assert result["wire_side_information_bytes"] == 414
    assert result["side_information_bytes"] == 414
    assert result["total_bytes"] == 1310
    assert result["exact_bpw"] == 5.1171875


def test_footprint_rejects_wrong_schedule_width_and_header_type():
    with pytest.raises(TrellisFormatError, match="for shape"):
        trellis_tensor_payload_breakdown(
            (2, 512),
            family=E2M1_FAMILY,
            body_rate_q256=512,
            layout=LAYOUT_FIXED_QUOTA,
            schedule=[2] * 256,
            alphabets={2: _alphabet(2, 4)},
        )
    with pytest.raises(TrellisFormatError, match="sidecar_header_bytes"):
        trellis_tensor_payload_breakdown(
            (2, 256),
            family=E2M1_FAMILY,
            body_rate_q256=512,
            layout=LAYOUT_FIXED_QUOTA,
            schedule=[2] * 256,
            alphabets={2: _alphabet(2, 4)},
            sidecar_header_bytes=True,
        )


def test_footprint_enforces_gridbook_v1_uint32_header_bounds():
    schedule = [1] * 256
    alphabet = {1: _alphabet(1, 8)}
    with pytest.raises(TrellisFormatError, match="unsigned 32-bit header"):
        trellis_tensor_payload_breakdown(
            ((1 << 32), 256),
            family=E4M3_FAMILY,
            body_rate_q256=256,
            layout=LAYOUT_FIXED_QUOTA,
            schedule=schedule,
            alphabets=alphabet,
        )
    with pytest.raises(TrellisFormatError, match="unsigned 32-bit header"):
        trellis_tensor_payload_breakdown(
            (1, (1 << 32)),
            family=E4M3_FAMILY,
            body_rate_q256=256,
            layout=LAYOUT_FIXED_QUOTA,
            schedule=schedule,
            alphabets=alphabet,
        )
    # rows itself fits uint32, but E4M3's four-byte-per-row scale_size field
    # would be 2**32 and cannot be serialized by Gridbook's v1 header.
    with pytest.raises(TrellisFormatError, match="scale plane.*uint32"):
        trellis_tensor_payload_breakdown(
            ((1 << 30), 256),
            family=E4M3_FAMILY,
            body_rate_q256=256,
            layout=LAYOUT_FIXED_QUOTA,
            schedule=schedule,
            alphabets=alphabet,
        )
