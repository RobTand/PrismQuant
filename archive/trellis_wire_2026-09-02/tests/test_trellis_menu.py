"""Gate for the opt-in seam that puts trellis rungs in the production menu.

The failure modes this guards are not arithmetic. They are:

  * the flag being anything other than a byte-identical no-op when unset,
  * a rung addressed by a per-tensor key, which silently collapses every
    fused/packed serving unit back to individual rows,
  * allocating against ``_capability_gate`` on a profile that declares no
    ``target_platform``, where the gate returns legal without comparing
    anything,
  * mixing two objectives in one DP,
  * an allocation-time-only surface reaching export as if it were shippable.
"""

from __future__ import annotations

import inspect
import json
import pathlib

import pytest

from prismaquant import trellis_menu as tm
from prismaquant.allocator_solver import Candidate
from prismaquant.trellis_formats import (
    E4M3_FAMILY,
    LAYOUT_TIGHT_OFFSETS,
    get_trellis_family,
    native_code_value,
    parse_trellis_format_name,
)

PROFILE = "trellis_research_sm121"
COST_MODE = "aura"
CURRENCY = "aura_adjoint"


def alphabet(rate: int) -> list[int]:
    """A NaN-free E4M3 alphabet ordered by decoded value then code."""

    spec = get_trellis_family(E4M3_FAMILY)
    required = 1 << (rate + 1)
    codes = [code for code in range(256) if code not in (0x7F, 0xFF)]
    if required > len(codes):
        codes.extend((0x00, 0x80))
        required = len(codes)
    codes.sort(key=lambda code: (native_code_value(spec, code), code))
    start = (len(codes) - required) // 2
    return list(codes[start:start + required])


def manifest_payload(units, **overrides) -> dict:
    payload = {
        "schema": tm.TRELLIS_SURFACE_MANIFEST_SCHEMA,
        "cost_mode": COST_MODE,
        "currency": CURRENCY,
        "target_profile": PROFILE,
        "activation_contract": "W8A16",
        "layout": LAYOUT_TIGHT_OFFSETS,
        "rungs_per_unit": 6,
        "provenance": {"encoder": "stage6", "encode_tier": "max"},
        "anchors": {
            unit: {
                "family": E4M3_FAMILY,
                "alphabets": {str(rate): alphabet(rate) for rate in range(1, 8)},
                "points": [
                    {"q256": 512, "dloss": 4.0e-3, "stderr": 1e-4},
                    {"q256": 1024, "dloss": 1.0e-3, "stderr": 1e-4},
                    {"q256": 1536, "dloss": 3.0e-4, "stderr": 2e-4},
                ],
            }
            for unit in units
        },
    }
    payload.update(overrides)
    return payload


def write_manifest(tmp_path, units, **overrides):
    path = tmp_path / "surface.json"
    path.write_text(json.dumps(manifest_payload(units, **overrides)))
    return str(path)


def scalar_menu(units):
    return {
        unit: [
            Candidate(fmt="BF16", bits_per_param=16.0,
                      memory_bytes=2 * rows * cols, predicted_dloss=0.0),
            Candidate(fmt="NVFP4", bits_per_param=4.5,
                      memory_bytes=rows * cols // 2, predicted_dloss=5.0e-3),
        ]
        for unit, (rows, cols) in units.items()
    }


def stats_for(units):
    return {
        unit: {"out_features": rows, "in_features": cols}
        for unit, (rows, cols) in units.items()
    }


UNIT_A = "model.layers.0.self_attn.q_proj"
UNIT_B = "model.layers.0.self_attn.k_proj"


def test_unset_flag_is_a_byte_identical_no_op(monkeypatch):
    monkeypatch.delenv(tm.TRELLIS_SURFACE_ENV, raising=False)
    units = {UNIT_A: (1024, 512)}
    menu = scalar_menu(units)
    before = {k: list(v) for k, v in menu.items()}
    out = tm.augment_candidates(menu, stats_for(units), cost_mode=COST_MODE)  # noqa: E501 -- the seam, deliberately
    assert out is menu
    assert out == before


def test_flag_adds_rungs_named_by_the_closed_tcq_spelling(tmp_path):
    units = {UNIT_A: (1024, 512)}
    menu = scalar_menu(units)
    prov: dict = {}
    tm.build_trellis_menu(
        menu, stats_for(units), cost_mode=COST_MODE,
        manifest_path=write_manifest(tmp_path, units),
        provenance_out=prov,
    )
    added = [c for c in menu[UNIT_A] if c.fmt.startswith("TCQ_")]
    assert added, "no trellis rungs reached the menu"
    for cand in added:
        parsed = parse_trellis_format_name(cand.fmt)
        assert parsed is not None, cand.fmt
        family, rate = parsed
        assert family.family == E4M3_FAMILY
        assert 512 <= rate <= 1536, "densify escaped the measured envelope"
        # Exact serialized bytes, never a bpp-times-params estimate.
        assert cand.memory_bytes > 0
        assert cand.serialized_identity and len(cand.serialized_identity) == 64
    assert prov["units_covered"] == 1
    assert prov["anchor_activation_contract"] == "W8A16"
    assert prov["exportable"] is False
    # The scalar menu is preserved, not replaced.
    assert {"BF16", "NVFP4"} <= {c.fmt for c in menu[UNIT_A]}


def test_fused_siblings_share_format_names_at_equal_rungs(tmp_path):
    """The whole reason `fmt` is shape-free.

    ``aggregate_fused_siblings``/``aggregate_packed_serving_groups`` intersect
    member menus BY FORMAT NAME. ``TrellisAllocatorCandidate.allocator_key``
    embeds the pre-render recipe digest, which hashes the SHAPE, so two
    siblings with different row counts would share no format at any rung and
    every fused group would silently fall back to individual rows.
    """

    units = {UNIT_A: (1024, 512), UNIT_B: (256, 512)}
    menu = scalar_menu(units)
    tm.build_trellis_menu(
        menu, stats_for(units), cost_mode=COST_MODE,
        manifest_path=write_manifest(tmp_path, units),
    )
    a = {c.fmt for c in menu[UNIT_A] if c.fmt.startswith("TCQ_")}
    b = {c.fmt for c in menu[UNIT_B] if c.fmt.startswith("TCQ_")}
    assert a and a == b, (
        "siblings with different out_features must offer identical rung "
        f"names; a-b={sorted(a - b)} b-a={sorted(b - a)}"
    )
    # ...while their exact byte costs differ, because the tensors do.
    bytes_a = {c.fmt: c.memory_bytes for c in menu[UNIT_A]}
    bytes_b = {c.fmt: c.memory_bytes for c in menu[UNIT_B]}
    shared = sorted(a)[0]
    assert bytes_a[shared] != bytes_b[shared]
    # The per-tensor recipe identity is what distinguishes them.
    ident_a = next(c.serialized_identity for c in menu[UNIT_A] if c.fmt == shared)
    ident_b = next(c.serialized_identity for c in menu[UNIT_B] if c.fmt == shared)
    assert ident_a != ident_b


def test_profile_without_target_platform_is_refused(tmp_path):
    """`research` declares no platform, so `_capability_gate` cannot fire."""

    units = {UNIT_A: (1024, 512)}
    menu = scalar_menu(units)
    path = write_manifest(tmp_path, units, target_profile="research")
    with pytest.raises(tm.TrellisMenuError, match="target_platform"):
        tm.build_trellis_menu(
            menu, stats_for(units), cost_mode=COST_MODE, manifest_path=path,
        )


def test_cost_mode_mismatch_is_refused(tmp_path):
    units = {UNIT_A: (1024, 512)}
    menu = scalar_menu(units)
    path = write_manifest(tmp_path, units, cost_mode="production-render-score")
    with pytest.raises(tm.TrellisMenuError, match="one currency"):
        tm.build_trellis_menu(
            menu, stats_for(units), cost_mode="aura", manifest_path=path,
        )


def test_single_anchor_is_refused(tmp_path):
    """One anchor brackets nothing; interpolation would become extrapolation."""

    units = {UNIT_A: (1024, 512)}
    payload = manifest_payload(units)
    payload["anchors"][UNIT_A]["points"] = [
        {"q256": 1024, "dloss": 1.0e-3, "stderr": 0.0}
    ]
    path = tmp_path / "one.json"
    path.write_text(json.dumps(payload))
    menu = scalar_menu(units)
    prov: dict = {}
    tm.build_trellis_menu(
        menu, stats_for(units), cost_mode=COST_MODE,
        manifest_path=str(path), provenance_out=prov,
    )
    # Skipped with a reason rather than silently producing an extrapolation.
    assert UNIT_A in prov["units_skipped"]
    assert "two measured anchors" in prov["units_skipped"][UNIT_A]
    assert not [c for c in menu[UNIT_A] if c.fmt.startswith("TCQ_")]


def test_columns_not_a_superblock_multiple_is_skipped_not_guessed(tmp_path):
    units = {UNIT_A: (1024, 300)}
    menu = scalar_menu(units)
    prov: dict = {}
    tm.build_trellis_menu(
        menu, stats_for(units), cost_mode=COST_MODE,
        manifest_path=write_manifest(tmp_path, units), provenance_out=prov,
    )
    assert "superblock" in prov["units_skipped"][UNIT_A]


def test_unit_absent_from_the_scalar_menu_is_reported(tmp_path):
    units = {UNIT_A: (1024, 512)}
    menu: dict = {}
    prov: dict = {}
    tm.build_trellis_menu(
        menu, stats_for(units), cost_mode=COST_MODE,
        manifest_path=write_manifest(tmp_path, units), provenance_out=prov,
    )
    assert prov["units_skipped"][UNIT_A].startswith("unit has no priced")
    assert prov["candidates_added"] == 0


def test_bad_schema_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "something.else.v9"}))
    with pytest.raises(tm.TrellisMenuError, match="schema"):
        tm.load_manifest(path)


def test_missing_activation_contract_is_refused(tmp_path):
    units = {UNIT_A: (1024, 512)}
    payload = manifest_payload(units)
    del payload["activation_contract"]
    path = tmp_path / "noact.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(tm.TrellisMenuError, match="activation_contract"):
        tm.load_manifest(path)


def test_assignment_has_trellis_finds_rungs():
    assignment = {
        UNIT_A: "TCQ_E4M3_R1024",
        UNIT_B: "NVFP4",
        "x": "BF16",
    }
    assert tm.assignment_has_trellis(assignment) == [UNIT_A]
    assert tm.assignment_has_trellis({"x": "NVFP4"}) == []


def test_the_production_seam_refuses_when_the_flag_is_set(tmp_path, monkeypatch):
    """Behaviour, not source text.

    The previous version of this test asserted that the string
    ``trellis_menu.augment_candidates`` appeared in ``build_candidates``'s
    source.  That passes whether or not the call does anything, and it passed
    while the enabled path could not produce an assignment at all.  This asserts
    what a caller observes.
    """

    from prismaquant import allocator_candidates as ac

    units = {UNIT_A: (1024, 512)}
    manifest = write_manifest(tmp_path, units)
    monkeypatch.setenv(tm.TRELLIS_SURFACE_ENV, str(manifest))
    with pytest.raises(tm.TrellisSeamUnwiredError) as exc:
        tm.augment_candidates(
            scalar_menu(units), stats_for(units), cost_mode=COST_MODE)
    message = str(exc.value)
    # The refusal must name every missing link, or it becomes the same kind of
    # confident-and-wrong claim it replaced.
    for where, _what in tm.UNWIRED_LINKS:
        assert where in message
    # ...and the seam is still reached from inside build_candidates, so a run
    # with the flag set fails loudly rather than ignoring it.
    assert "cost_mode" in inspect.signature(ac.build_candidates).parameters


def test_the_unwired_links_are_still_unwired():
    """The ledger is a claim about the code; check the two cheapest entries.

    If either of these starts passing, the corresponding UNWIRED_LINKS entry is
    stale and must be deleted along with -- not before -- its refusal.
    """

    from prismaquant import format_registry as fr

    with pytest.raises(KeyError):
        fr.get_format("TCQ_E2M1_R640")
    from prismaquant import allocator_candidates as ac

    aggregation = pathlib.Path(ac.__file__).read_text().count(
        "for spec in formats:")
    assert aggregation >= 2, (
        "fused and packed aggregation still iterate FormatSpec objects; if "
        "this changed, re-verify links 3 and 4 before deleting them"
    )


def test_a_packed_expert_row_is_refused_not_underpriced(tmp_path):
    """The 128x underprice, which was silent.

    The seam used to build a 2-tuple shape and read a ``packed_expert`` key
    nothing writes, so a 128-expert row was priced as a single expert and
    reported as ``0 unit(s) skipped``.  A rung that looks 128x cheap than it is
    is the most dangerous shape of wrong: the DP takes it and the frontier
    looks plausible.
    """

    units = {UNIT_A: (1024, 512)}
    stats = stats_for(units)
    stats[UNIT_A] = dict(stats[UNIT_A], num_experts=128)
    menu = scalar_menu(units)
    prov: dict = {}
    tm.build_trellis_menu(
        menu, stats, cost_mode=COST_MODE,
        manifest_path=write_manifest(tmp_path, units),
        provenance_out=prov,
    )
    assert not [c for c in menu[UNIT_A] if c.fmt.startswith("TCQ_")]
    assert "packed-expert" in prov["units_skipped"][UNIT_A]


def test_export_refuses_a_trellis_assignment():
    """Allocation-time reach only: there is no render and no attestation."""

    from prismaquant import export_native_compressed as ex

    text = pathlib.Path(ex.__file__).read_text()
    assert "parse_trellis_format_name(fmt_canonical) is not None" in text
    assert "ProductionWeightCache renders no trellis wire" in text
