"""Gate for the opt-in seam that puts trellis rungs in the production menu.

The failure modes this guards are not arithmetic. They are:

  * the flag being anything other than a byte-identical no-op when unset,
  * a rung addressed by a per-tensor key, which silently collapses every
    fused/packed serving unit back to individual rows,
  * a rung dropped from a coupled unit's menu because aggregation iterated
    the REGISTRY instead of the members' candidate lists -- silent, and the
    reason the seam refused as a whole until 2026-08-29,
  * allocating against ``_capability_gate`` on a profile that declares no
    ``target_platform``, where the gate returns legal without comparing
    anything,
  * mixing two objectives in one DP,
  * a byte path falling through to a closed form that cannot express a
    trellis wire, and
  * an allocation-time-only surface reaching export as if it were shippable.

Every test here asserts BEHAVIOUR. Two earlier tests in this file asserted
source text -- one counted ``"for spec in formats:"`` occurrences, one
grepped the exporter -- and both passed while the enabled seam could not
produce an assignment at all. They are replaced below by tests that call the
code.
"""

from __future__ import annotations

import json

import pytest

from prismaquant import trellis_menu as tm
from prismaquant.allocator_solver import Candidate, promote_serving_units
from prismaquant.trellis_formats import (
    E4M3_FAMILY,
    LAYOUT_TIGHT_OFFSETS,
    get_trellis_family,
    native_code_value,
    parse_trellis_format_name,
)

PROFILE = "trellis_research_sm121"
COST_MODE = "aura"
#: The objective ``COST_MODE=aura`` prices in. Spelled exactly as
#: ``run-pipeline.sh`` resolves ``COST_OBJECTIVE``; the seam refuses anything
#: else, which is the surviving UNWIRED_LINKS entry.
CURRENCY = "aura-adjoint"
#: What the measured trellis ladder ACTUALLY denominates its anchors in.
LADDER_CURRENCY = "weighted_sse_activation_second_moment"


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
        unit: {
            "out_features": rows,
            "in_features": cols,
            "n_params": rows * cols,
            "h_trace": 0.5,
        }
        for unit, (rows, cols) in units.items()
    }


def costs_for(units):
    return {
        unit: {
            "BF16": {"weight_mse": 0.0, "predicted_dloss": 0.0},
            "NVFP4": {"weight_mse": 0.02, "predicted_dloss": 5.0e-3},
        }
        for unit in units
    }


class _QKVProfile:
    def fused_sibling_group(self, name: str) -> str | None:
        if name.endswith((".q_proj", ".k_proj", ".v_proj")):
            return name.rsplit(".", 1)[0] + ".qkv_proj"
        return None


UNIT_A = "model.layers.0.self_attn.q_proj"
UNIT_B = "model.layers.0.self_attn.k_proj"


# ---------------------------------------------------------------------------
# The unset flag
# ---------------------------------------------------------------------------
def test_unset_flag_is_a_byte_identical_no_op(monkeypatch):
    monkeypatch.delenv(tm.TRELLIS_SURFACE_ENV, raising=False)
    units = {UNIT_A: (1024, 512)}
    menu = scalar_menu(units)
    stats = stats_for(units)
    before = {k: list(v) for k, v in menu.items()}
    out = tm.augment_candidates(menu, stats, cost_mode=COST_MODE)  # the seam
    assert out is menu
    assert out == before
    # and nothing was recorded on the stats either
    assert "_memory_bytes_by_format" not in stats[UNIT_A]


# ---------------------------------------------------------------------------
# The built menu
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# The refusals the manifest exists to make possible
# ---------------------------------------------------------------------------
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


def test_the_ladders_own_currency_is_the_surviving_refusal(tmp_path):
    """UNWIRED_LINKS[0], and the only one left.

    A surface whose anchors were measured in the ladder's weighted SSE -- an
    output-MSE proxy -- cannot be ranked against AURA-priced NVFP4 rungs. No
    plumbing fixes that, so it is a refusal rather than a link, and the
    message must name where the currency comes from.
    """

    units = {UNIT_A: (1024, 512)}
    menu = scalar_menu(units)
    path = write_manifest(tmp_path, units, currency=LADDER_CURRENCY)
    with pytest.raises(tm.TrellisSeamUnwiredError) as exc:
        tm.build_trellis_menu(
            menu, stats_for(units), cost_mode=COST_MODE, manifest_path=path,
        )
    message = str(exc.value)
    assert LADDER_CURRENCY in message
    assert "aura-adjoint" in message
    for where, _what in tm.UNWIRED_LINKS:
        assert where in message
    # Nothing was added on the way to the refusal.
    assert not [c for c in menu[UNIT_A] if c.fmt.startswith("TCQ_")]


def test_the_only_remaining_unwired_link_is_the_currency():
    """The ledger is a claim about the code; seven entries are gone."""

    assert [where for where, _ in tm.UNWIRED_LINKS] == [
        "trellis_rate_surface.py:43-52"
    ]


def test_an_unstamped_cost_table_is_refused_not_defaulted(tmp_path):
    """re-vet R2: the run's objective is attested, or the surface is refused.

    Comparing against ``os.environ.get('COST_MODE', 'aura')`` was the old
    behaviour and it compared against a default no run ever exported.
    """

    units = {UNIT_A: (1024, 512)}
    menu = scalar_menu(units)
    path = write_manifest(tmp_path, units, cost_mode="")
    with pytest.raises(tm.TrellisMenuError, match=r"provenance\['cost_mode'\]"):
        tm.build_trellis_menu(
            menu, stats_for(units), cost_mode="", manifest_path=path,
        )


def test_a_cost_mode_with_no_declared_objective_is_refused(tmp_path):
    units = {UNIT_A: (1024, 512)}
    menu = scalar_menu(units)
    path = write_manifest(tmp_path, units, cost_mode="grouped-kl")
    with pytest.raises(tm.TrellisMenuError, match="names no objective"):
        tm.build_trellis_menu(
            menu, stats_for(units), cost_mode="grouped-kl", manifest_path=path,
        )


# ---------------------------------------------------------------------------
# Skips that must be counted, not silent
# ---------------------------------------------------------------------------
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


def test_a_packed_expert_row_is_refused_not_underpriced(tmp_path):
    """The 128x underprice, which was silent.

    The seam used to build a 2-tuple shape and read a ``packed_expert`` key
    nothing writes, so a 128-expert row was priced as a single expert and
    reported as ``0 unit(s) skipped``.  A rung that looks 128x cheaper than it
    is is the most dangerous shape of wrong: the DP takes it and the frontier
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


# ---------------------------------------------------------------------------
# Link 1: no TCQ FormatSpec, on purpose -- the bytes ride the stats instead
# ---------------------------------------------------------------------------
def test_no_tcq_name_is_a_formatspec_and_that_is_the_design():
    """A registered spec could only be plausible and WRONG.

    ``FormatSpec.memory_bytes_for_shape`` is a closed form over weight/scale
    bits; a rung's exact size needs the layout, the per-column schedule and
    the alphabet directory the manifest declares. So the registry keeps
    refusing -- but ``canonical_format_name`` must still pass the name
    through unchanged, or every byte path would look up a mangled key.
    """

    from prismaquant import format_registry as fr

    with pytest.raises(KeyError):
        fr.get_format("TCQ_E2M1_R640")
    assert fr.canonical_format_name("TCQ_E2M1_R640") == "TCQ_E2M1_R640"


def test_the_seam_records_exact_bytes_where_every_byte_path_reads_them(tmp_path):
    """Links 1/2/6, at the source: ``_memory_bytes_by_format``.

    ``build_candidates`` writes this map for every FormatSpec row and the
    allocator's payload filter, ``footprint``, ``compute_achieved``,
    ``kl_measurement`` and bit attribution all PREFER it over the registry.
    Writing it from the seam is what wires all of them at once.
    """

    units = {UNIT_A: (1024, 512)}
    menu = scalar_menu(units)
    stats = stats_for(units)
    tm.build_trellis_menu(
        menu, stats, cost_mode=COST_MODE,
        manifest_path=write_manifest(tmp_path, units),
    )
    recorded = stats[UNIT_A]["_memory_bytes_by_format"]
    rungs = [c for c in menu[UNIT_A] if c.fmt.startswith("TCQ_")]
    assert rungs
    for cand in rungs:
        assert recorded[cand.fmt] == int(cand.memory_bytes)
    # The per-tensor recipe identity is recorded alongside, same as CB.
    identities = stats[UNIT_A]["_serialized_identity_by_format"]
    assert identities[rungs[0].fmt] == rungs[0].serialized_identity


def test_footprint_prices_a_trellis_row_from_the_recorded_bytes(tmp_path):
    """Link 6: the byte-budget path, which has its own registry lookup."""

    from prismaquant import footprint as fp

    units = {UNIT_A: (1024, 512)}
    menu = scalar_menu(units)
    stats = stats_for(units)
    tm.build_trellis_menu(
        menu, stats, cost_mode=COST_MODE,
        manifest_path=write_manifest(tmp_path, units),
    )
    rung = next(c for c in menu[UNIT_A] if c.fmt.startswith("TCQ_"))
    info = fp.assignment_artifact_bytes(
        {UNIT_A: rung.fmt},
        stats,
        source_total_bytes=10_000_000,
        source_manifest=None,
        regime="bf16",
    )
    assert info["body_quant_bytes"] == int(rung.memory_bytes)


def test_footprint_refuses_a_trellis_row_it_has_no_measured_bytes_for():
    """No closed-form fallback: a missing map is refused, not approximated."""

    from prismaquant import footprint as fp

    stats = {UNIT_A: {"out_features": 1024, "in_features": 512,
                      "n_params": 1024 * 512}}
    with pytest.raises(ValueError, match="_memory_bytes_by_format"):
        fp.assignment_artifact_bytes(
            {UNIT_A: "TCQ_E4M3_R1024"},
            stats,
            source_total_bytes=10_000_000,
            source_manifest=None,
            regime="bf16",
        )


# ---------------------------------------------------------------------------
# Links 3 and 4: super-item aggregation over MEMBER menus
# ---------------------------------------------------------------------------
def test_a_fused_super_item_offers_the_rung_its_members_share(tmp_path):
    """Link 3, end to end through the real seam.

    Before this, ``aggregate_fused_siblings`` built each super item's menu by
    iterating FormatSpec OBJECTS, so a rung both members offered was dropped
    from the group with no error -- on a dense model that left only
    ``o_proj`` and ``down_proj`` able to hold one.
    """

    from prismaquant import format_registry as fr
    from prismaquant.allocator_candidates import (
        _FUSED_SIBLING_MARKER,
        aggregate_fused_siblings,
    )

    units = {UNIT_A: (1024, 512), UNIT_B: (256, 512)}
    menu = scalar_menu(units)
    stats = stats_for(units)
    tm.build_trellis_menu(
        menu, stats, cost_mode=COST_MODE,
        manifest_path=write_manifest(tmp_path, units),
    )
    shared = sorted(
        ({c.fmt for c in menu[UNIT_A]} & {c.fmt for c in menu[UNIT_B]})
        - {"BF16", "NVFP4"}
    )
    assert shared, "the fixture must give the siblings a shared rung"

    specs = [fr.REGISTRY["NVFP4"], fr.REGISTRY["BF16"]]
    stats_ext, _costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs_for(units), specs, menu, _QKVProfile())
    super_name = next(n for n in cands_ext if _FUSED_SIBLING_MARKER in n)
    offered = {c.fmt: c for c in cands_ext[super_name]}
    for fmt in shared:
        assert fmt in offered, f"{fmt} was dropped from the fused group"
        # Exact sums of the members', not a re-derived estimate.
        assert offered[fmt].memory_bytes == sum(
            next(c.memory_bytes for c in menu[u] if c.fmt == fmt)
            for u in (UNIT_A, UNIT_B)
        )
        assert offered[fmt].predicted_dloss == pytest.approx(sum(
            next(c.predicted_dloss for c in menu[u] if c.fmt == fmt)
            for u in (UNIT_A, UNIT_B)
        ))
        # And the exact bytes are recorded where the byte paths read them.
        assert (stats_ext[super_name]["_memory_bytes_by_format"][fmt]
                == offered[fmt].memory_bytes)
    # The registry menu still comes FIRST, so an all-registry run is
    # unchanged (tests/test_super_item_menu_byte_identity.py pins that).
    assert [c.fmt for c in cands_ext[super_name]][:2] == ["NVFP4", "BF16"]


class _PackedProfile:
    def packed_expert_format_group(self, name: str) -> str | None:
        if ".experts." not in name:
            return None
        return name.split(".experts.")[0] + ".experts"


def test_a_packed_group_offers_a_registry_free_format_its_members_share():
    """Link 4, by injection.

    The seam REFUSES packed-expert rows (no per-expert trellis render
    exists), so no manifest can put a rung on a packed group today. The
    aggregation contract is still the thing under test, and it is exercised
    with synthetic registry-free candidates: what must not happen is a
    format every member offers vanishing from the group's menu.
    """

    from prismaquant import format_registry as fr
    from prismaquant.allocator_candidates import (
        _PACKED_GROUP_MARKER,
        aggregate_packed_serving_groups,
    )

    members = [
        f"model.layers.0.mlp.experts.{i}.{leaf}"
        for i in range(2) for leaf in ("w1", "w2")
    ]
    stats = {m: {"n_params": 4096, "h_trace": 0.3,
                 "out_features": 64, "in_features": 64} for m in members}
    costs = {m: {"BF16": {"weight_mse": 0.0, "predicted_dloss": 0.0},
                 "NVFP4": {"weight_mse": 0.02, "predicted_dloss": 1e-3}}
             for m in members}
    fake = "TCQ_E4M3_R1024"
    cands = {
        m: [
            Candidate(fmt="BF16", bits_per_param=16.0,
                      memory_bytes=8192, predicted_dloss=0.0),
            Candidate(fmt="NVFP4", bits_per_param=4.5,
                      memory_bytes=2304, predicted_dloss=1e-3),
            Candidate(fmt=fake, bits_per_param=4.0,
                      memory_bytes=2048, predicted_dloss=7e-4),
        ]
        for m in members
    }
    specs = [fr.REGISTRY["NVFP4"], fr.REGISTRY["BF16"]]
    stats_ext, _costs_ext, cands_ext = aggregate_packed_serving_groups(
        stats, costs, specs, cands, _PackedProfile())
    super_name = next(n for n in cands_ext if _PACKED_GROUP_MARKER in n)
    offered = {c.fmt: c for c in cands_ext[super_name]}
    assert fake in offered, "registry-free rung dropped from the packed group"
    assert offered[fake].memory_bytes == 2048 * len(members)
    assert offered[fake].predicted_dloss == pytest.approx(7e-4 * len(members))
    assert stats_ext[super_name]["_memory_bytes_by_format"][fake] == \
        2048 * len(members)
    assert [c.fmt for c in cands_ext[super_name]] == ["NVFP4", "BF16", fake]


def test_a_registry_free_format_only_some_members_offer_is_not_aggregated():
    """The intersection is still the contract: no member may be unpriced."""

    from prismaquant import format_registry as fr
    from prismaquant.allocator_candidates import (
        _PACKED_GROUP_MARKER,
        aggregate_packed_serving_groups,
    )

    members = ["model.layers.0.mlp.experts.0.w1",
               "model.layers.0.mlp.experts.1.w1"]
    stats = {m: {"n_params": 4096, "h_trace": 0.3} for m in members}
    costs = {m: {"BF16": {"weight_mse": 0.0, "predicted_dloss": 0.0}}
             for m in members}
    cands = {
        members[0]: [
            Candidate(fmt="BF16", bits_per_param=16.0,
                      memory_bytes=8192, predicted_dloss=0.0),
            Candidate(fmt="TCQ_E4M3_R1024", bits_per_param=4.0,
                      memory_bytes=2048, predicted_dloss=7e-4),
        ],
        members[1]: [
            Candidate(fmt="BF16", bits_per_param=16.0,
                      memory_bytes=8192, predicted_dloss=0.0),
        ],
    }
    specs = [fr.REGISTRY["BF16"]]
    _stats_ext, _costs_ext, cands_ext = aggregate_packed_serving_groups(
        stats, costs, specs, cands, _PackedProfile())
    super_name = next(n for n in cands_ext if _PACKED_GROUP_MARKER in n)
    assert [c.fmt for c in cands_ext[super_name]] == ["BF16"]


# ---------------------------------------------------------------------------
# Link 5: promotion can rank a selected rung
# ---------------------------------------------------------------------------
def test_promotion_ranks_a_trellis_rung_by_its_exact_serialized_rate():
    """Link 5 lives or dies on format_rank covering the MENU."""

    from prismaquant.allocator import _extend_format_rank_with_candidate_menu

    stats = {UNIT_A: {"n_params": 1024 * 512},
             UNIT_B: {"n_params": 256 * 512}}
    # 4.0 bits/param exactly: below NVFP4 (4.5) and BF16 (16.0).
    cands = {
        unit: [
            Candidate(fmt="NVFP4", bits_per_param=4.5,
                      memory_bytes=int(stats[unit]["n_params"] * 4.5 / 8),
                      predicted_dloss=1e-3),
            Candidate(fmt="BF16", bits_per_param=16.0,
                      memory_bytes=stats[unit]["n_params"] * 2,
                      predicted_dloss=0.0),
            Candidate(fmt="TCQ_E4M3_R1024", bits_per_param=4.0,
                      memory_bytes=int(stats[unit]["n_params"] * 4.0 / 8),
                      predicted_dloss=2e-3),
        ]
        for unit in (UNIT_A, UNIT_B)
    }
    rank, rates = _extend_format_rank_with_candidate_menu(
        {"NVFP4": 0, "BF16": 1}, {"NVFP4": 4.5, "BF16": 16.0}, stats, cands)
    assert rates["TCQ_E4M3_R1024"] == pytest.approx(4.0)
    # Ranked BELOW NVFP4 because it really is fewer bytes on this model.
    assert rank["TCQ_E4M3_R1024"] < rank["NVFP4"] < rank["BF16"]

    # ...and promotion now runs instead of raising KeyError, lifting the
    # cheaper sibling onto the unit's max-rank format.
    out = promote_serving_units(
        {UNIT_A: "TCQ_E4M3_R1024", UNIT_B: "NVFP4"},
        rank,
        profile=_QKVProfile(),
        include_moe=False,
    )
    assert out == {UNIT_A: "NVFP4", UNIT_B: "NVFP4"}


def test_an_all_registry_menu_leaves_the_rank_table_untouched():
    """Principle 6: the extension is a no-op when nothing needs it."""

    from prismaquant.allocator import _extend_format_rank_with_candidate_menu

    stats = {UNIT_A: {"n_params": 4096}}
    cands = {UNIT_A: [Candidate(fmt="NVFP4", bits_per_param=4.5,
                                memory_bytes=2304, predicted_dloss=1e-3)]}
    original = {"NVFP4": 0, "BF16": 1}
    rank, rates = _extend_format_rank_with_candidate_menu(
        original, {"NVFP4": 4.5, "BF16": 16.0}, stats, cands)
    assert rank == original
    assert rates == {"NVFP4": 4.5, "BF16": 16.0}


def test_promotion_names_the_rank_table_when_a_format_is_unranked():
    """A bare KeyError reads as a corrupt assignment; it is a rank-table bug."""

    with pytest.raises(KeyError, match="CANDIDATE MENU"):
        promote_serving_units(
            {UNIT_A: "TCQ_E4M3_R1024", UNIT_B: "NVFP4"},
            {"NVFP4": 0, "BF16": 1},
            profile=_QKVProfile(),
            include_moe=False,
        )


# ---------------------------------------------------------------------------
# Links 2 and 7: the assignment travels, and export refuses it
# ---------------------------------------------------------------------------
def test_layer_config_round_trips_a_trellis_assignment():
    """Link 2's tail: the exporter's pointed refusal must be REACHABLE.

    A generic "unsupported format string" in ``canonicalize_format`` would
    turn a research assignment into a parse error and hide the one message
    that explains why there is no export path.
    """

    from prismaquant.layer_config import (
        LAYER_CONFIG_META_KEY,
        canonicalize_assignment,
        canonicalize_format,
    )
    from prismaquant.schemas import validate_layer_config_payload

    payload = {
        UNIT_A: "TCQ_E4M3_R1024",
        UNIT_B: {"data_type": "nv_fp", "bits": 4},
        LAYER_CONFIG_META_KEY: {
            "schema": "prismaquant.layer_config_meta.v1",
            "trellis_surface": {"manifest_sha256": "0" * 64},
        },
    }
    validate_layer_config_payload(payload, "<test>")
    assert canonicalize_format("TCQ_E4M3_R1024") == "TCQ_E4M3_R1024"
    assert canonicalize_format("tcq_e4m3_r1024") == "TCQ_E4M3_R1024"
    assert canonicalize_assignment(payload) == {
        UNIT_A: "TCQ_E4M3_R1024", UNIT_B: "NVFP4",
    }


def test_an_unparseable_tcq_spelling_is_still_refused():
    from prismaquant.layer_config import canonicalize_format

    with pytest.raises(ValueError, match="unsupported format string"):
        canonicalize_format("TCQ_E9M9_R1024")


def test_export_refuses_a_trellis_assignment():
    """Allocation-time reach only: there is no render and no attestation.

    Behaviour, not a grep: the previous version of this test asserted two
    string literals appeared in the exporter's source, which would pass even
    if the branch were unreachable.
    """

    from prismaquant import export_native_compressed as ex

    with pytest.raises(ValueError) as exc:
        ex._coerce_runtime_legal_assignment(
            "/nonexistent-source-model",
            {UNIT_A: "TCQ_E4M3_R1024"},
            None,
        )
    message = str(exc.value)
    assert "renders no trellis wire" in message
    assert "activation-contract" in message


def test_the_provenance_carries_the_manifest_identity_and_the_run_currency(
        tmp_path):
    """Link 7: what must travel WITH the assignment (principles 12 and 14)."""

    units = {UNIT_A: (1024, 512)}
    menu = scalar_menu(units)
    prov: dict = {}
    path = write_manifest(tmp_path, units)
    tm.build_trellis_menu(
        menu, stats_for(units), cost_mode=COST_MODE,
        manifest_path=path, provenance_out=prov,
    )
    assert len(prov["manifest_sha256"]) == 64
    assert prov["manifest_sha256"] == tm.load_manifest(path).sha256
    assert prov["currency"] == CURRENCY
    assert prov["run_objective_currency"] == CURRENCY
    assert prov["anchor_activation_contract"] == "W8A16"
    assert prov["research_only"] is True
    assert prov["exportable"] is False
    # Rewriting the manifest changes the identity, which is the point of
    # hashing bytes rather than recording a path.
    payload = manifest_payload(units)
    payload["rungs_per_unit"] = 5
    (tmp_path / "surface.json").write_text(json.dumps(payload))
    assert tm.load_manifest(path).sha256 != prov["manifest_sha256"]


def test_the_production_seam_installs_the_menu_when_the_flag_is_set(
        tmp_path, monkeypatch):
    """The seam itself, through ``build_candidates`` -- behaviour, not source.

    An earlier test asserted the string ``trellis_menu.augment_candidates``
    appeared in ``build_candidates``'s source. That passes whether or not the
    call does anything, and it did pass while the enabled path could not
    produce an assignment at all.
    """

    from prismaquant import format_registry as fr
    from prismaquant.allocator_candidates import build_candidates

    units = {UNIT_A: (1024, 512)}
    monkeypatch.setenv(
        tm.TRELLIS_SURFACE_ENV, write_manifest(tmp_path, units))
    stats = stats_for(units)
    prov: dict = {}
    out = build_candidates(
        stats, costs_for(units),
        [fr.REGISTRY["NVFP4"], fr.REGISTRY["BF16"]],
        cost_mode=COST_MODE,
        trellis_provenance=prov,
    )
    rungs = [c for c in out[UNIT_A] if c.fmt.startswith("TCQ_")]
    assert rungs, "the seam did not install the menu"
    assert prov["candidates_added"] == len(rungs)
    assert stats[UNIT_A]["_memory_bytes_by_format"][rungs[0].fmt] == \
        int(rungs[0].memory_bytes)
