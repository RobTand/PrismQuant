"""The three gates a Tessera rung passes before the DP can see it.

Tessera's rate axis is continuous at a 1/256-bpp quantum, so "the menu" is not
a list somebody typed -- it is whatever three independent gates agree on:

* the **wire** can serialise these bytes (Tessera's own grammar),
* the **shape** tiles, including the shard a declared tensor-parallel degree
  hands the runtime,
* the pinned **runtime** attests the route (or the run explicitly opts into a
  research menu that says it does not).

These tests pin each gate separately, pin that the reductions applied
afterwards are the exact ones (dominance, never a hull), and pin that a menu
with no Tessera rung on it behaves exactly as it did before any of this
existed.
"""
import hashlib
from fractions import Fraction

import pytest

from prismaquant import format_registry as fr
from prismaquant import tessera_menu as tm
from prismaquant import tessera_runtime_contract as trc
from prismaquant.tessera_formats import (
    TesseraFormatError, format_promotion_class, parse_tessera_format_name,
)

SHAPE = (2048, 1024)


@pytest.fixture
def dev_pin(monkeypatch):
    """Turn on the Tessera development pin for one test.

    Never module-scoped and never autouse: the default-menu test asserts the
    menu is EMPTY without it, and a leaked pin would silently invert exactly
    the assertion that says production is still fail-closed.
    """
    monkeypatch.setenv(trc.TESSERA_DEV_PIN_ENV, trc.TESSERA_DEV_PIN_COMMIT)
    return trc.TESSERA_DEV_PIN_COMMIT


# ---------------------------------------------------------------------------
# Gate 1: the families, and that they are Tessera's answer and not ours
# ---------------------------------------------------------------------------

def test_families_are_the_serialisable_set_tessera_declares():
    names = sorted(f.name for f in tm.menu_families())
    assert names == [
        "TESSERA_BF16_K1", "TESSERA_E2M1_K1", "TESSERA_E2M1_K2",
        "TESSERA_E4M3_K1",
    ], names
    # BF16 is the fourth because Tessera admitted it to SERIALISABLE_GRIDS and
    # the anchor budget stopped reading a WINDOW grid as a TCQ forest.  It is
    # here to be PRICED, not to be served: no pinned runtime attests a Tessera
    # BF16 route (Tessera issue #9), so the attested menu below still offers
    # exactly two rungs and none of them is this family's.


def test_e4m3_arity_2_is_refused_by_tessera_not_by_us():
    """The brief listed E4M3 K1/K2; the encoder refuses K2 and the code wins.

    Pinned as a test rather than a comment because the refusal is a COST
    refusal (2**16 anchors scored per trellis step), so it would come back the
    day the anchor budget moves -- and the menu would silently widen.
    """
    from prismaquant.tessera_formats import tessera_family

    with pytest.raises(TesseraFormatError, match="anchors scored"):
        tessera_family("E4M3", 2)


# ---------------------------------------------------------------------------
# Gate 2: shape, and the tensor-parallel shard of it
# ---------------------------------------------------------------------------

def test_shard_granularity_matches_a_real_encoded_unit():
    """The menu's granularity is Tessera's own, on a unit Tessera encoded.

    ``tessera_menu`` cannot encode 3000 rungs to ask where they cut, so it
    hands ``tessera.layout.shard_granularity`` a ``_ShardGeometry`` carrying
    the geometry an encode *would* produce. Three of the fields that object
    supplies are read there through ``getattr(unit, name, default)``, so an
    omitted or mis-set one answers confidently for a different wire. This
    encodes real units -- through the exporter's own ``encode_linear_planes``,
    the call that writes the bytes -- under every plane and body the wire
    recipe actually writes, and requires the two answers to be equal.

    That is LUT16 and CHANNEL, not all three planes: no rung of any
    serialisable family resolves to an S6b plane under ``wire_recipe``
    (E2M1_K1 and E2M1_K2 are LUT16 throughout, E4M3_K1 is CHANNEL
    throughout), so an S6b arm here would pin a wire nothing writes.
    Both bodies do occur -- the E2M1x2 cap is TCQ, everything below it
    is the window body.
    """
    from tessera.export import encode_linear_planes
    from tessera.layout import shard_granularity as tessera_granularity

    import torch

    from prismaquant.tessera_formats import get_tessera_family

    seen_planes = set()
    seen_bodies = set()
    cases = [
        ("TESSERA_E2M1_K1", 512, (64, 512)),
        ("TESSERA_E2M1_K1", 448, (64, 512)),
        ("TESSERA_E2M1_K2", 896, (64, 512)),
        ("TESSERA_E2M1_K2", 700, (64, 512)),
        ("TESSERA_E4M3_K1", 1024, (64, 512)),
        ("TESSERA_E4M3_K1", 1023, (64, 512)),
        ("TESSERA_E4M3_K1", 900, (64, 512)),
    ]
    generator = torch.Generator().manual_seed(0)
    for family_name, rung, shape in cases:
        spec = get_tessera_family(family_name)
        weight = torch.randn(shape, generator=generator, dtype=torch.float32)
        _exported, unit, _forests = encode_linear_planes(
            weight, grid=spec.payload_grid(), q256=rung,
            name=f"{family_name}_R{rung}", verify=False,
        )
        seen_planes.add(int(unit.scale_plane))
        seen_bodies.add(int(unit.body))
        theirs = tessera_granularity(unit, tm.SUPERBLOCK_WEIGHTS, spec.arity)
        ours = tm.tessera_shard_granularity(family_name, rung, shape)
        assert ours == tuple(int(x) for x in theirs), (family_name, rung, ours, theirs)
    from tessera.manifest import BodyKind, ScalePlaneKind

    assert seen_planes == {int(ScalePlaneKind.LUT), int(ScalePlaneKind.CHANNEL)}, (
        f"planes exercised: {sorted(seen_planes)}")
    assert seen_bodies == {int(BodyKind.TCQ), int(BodyKind.WINDOW)}, (
        f"bodies exercised: {sorted(seen_bodies)}")


def test_a_mixed_schedule_raises_the_column_granularity_to_the_superblock():
    """The fact a local derivation got wrong, kept as a test.

    Until Tessera published ``shard_granularity`` this module derived the
    column period from the scale plane alone and answered ``1`` for every
    CHANNEL-plane rung. Tessera's own derivation says a *mixed* Bresenham
    schedule only closes its quota on a whole superblock, so all but the
    handful of integer-rate rungs cut on 256 columns -- which is the binding
    TP constraint for every row-parallel Linear on this menu.
    """
    integer_rate = tm.tessera_shard_granularity("TESSERA_E4M3_K1", 1024, SHAPE)
    mixed_rate = tm.tessera_shard_granularity("TESSERA_E4M3_K1", 1023, SHAPE)
    assert integer_rate == (1, 1), integer_rate
    assert mixed_rate == (1, tm.SUPERBLOCK_WEIGHTS), mixed_rate


def test_the_tp_gate_asks_tessera_about_the_axis_vllm_actually_shards():
    """``PARALLEL_COLUMN`` is vLLM's word for a cut Tessera calls ``row``.

    vLLM's ColumnParallelLinear splits the OUTPUT features; those are the
    unit's rows, and ``tessera.layout.can_shard`` takes ``axis="row"`` for
    them. Inverting the pair answers plausibly in both directions and gates
    exactly the wrong half of a model, so it is pinned against a rung whose
    two granularities differ by a factor of 256.
    """
    from tessera.layout import can_shard

    from prismaquant.tessera_formats import get_tessera_family
    from prismaquant.tessera_menu import _shard_geometry

    family, shape, rung = "TESSERA_E4M3_K1", (2048, 4096), 1023
    spec = get_tessera_family(family)
    geometry = _shard_geometry(family, rung, *shape)
    assert tm.tessera_shard_granularity(family, rung, shape) == (1, 256)

    # 4096 columns cut 32 ways is 128, under the 256-column period; 2048 rows
    # cut 32 ways is 64, a multiple of the 1-row period. So the two axes give
    # opposite answers at tp=32, which is what makes the mapping testable.
    assert can_shard(geometry, 32, "row", tm.SUPERBLOCK_WEIGHTS, spec.arity)
    assert not can_shard(geometry, 32, "column", tm.SUPERBLOCK_WEIGHTS, spec.arity)

    legal_col, _ = tm.tessera_tp_legal(
        family, rung, shape, tp_degree=32, parallel_kind=tm.PARALLEL_COLUMN)
    legal_row, reason_row = tm.tessera_tp_legal(
        family, rung, shape, tp_degree=32, parallel_kind=tm.PARALLEL_ROW)
    assert legal_col, "a column-parallel cut of the rows is legal here"
    assert not legal_row, "a row-parallel cut of the columns is not"
    assert "column_granularity" in reason_row, reason_row


def test_tp_degree_is_a_per_unit_legality_input_and_knows_the_direction():
    """TP legality is not one number: it depends on WHICH axis is sharded.

    A Tessera rate is realised as a per-column Bresenham schedule over the
    reduce dimension, so what a rank can encode is a function of ITS column
    count. Column-parallel Linears (q/k/v/gate/up) shard ``out_features`` and
    keep every column, so their realisable set is unchanged at any TP degree;
    row-parallel ones (o_proj/down_proj) shard ``in_features``, and at TP=8
    a 1024-column unit becomes a 128-column one, which is a strictly smaller
    realisable set. A gate that took only a degree and not a direction would
    get one of these two wrong.
    """
    mode = tm.MENU_RESEARCH
    col_1 = tm.expand_tessera_menu(
        SHAPE, mode=mode, tp_degree=1, parallel_kind=tm.PARALLEL_COLUMN)
    col_8 = tm.expand_tessera_menu(
        SHAPE, mode=mode, tp_degree=8, parallel_kind=tm.PARALLEL_COLUMN)
    assert len(col_8) == len(col_1), (len(col_8), len(col_1))

    row_1 = tm.expand_tessera_menu(
        SHAPE, mode=mode, tp_degree=1, parallel_kind=tm.PARALLEL_ROW)
    row_8 = tm.expand_tessera_menu(
        SHAPE, mode=mode, tp_degree=8, parallel_kind=tm.PARALLEL_ROW)
    assert len(row_8) < len(row_1), (len(row_8), len(row_1))
    assert row_8, "TP=8 must not empty the menu on a shape that shards"


def test_tp_gate_refuses_a_shard_that_does_not_divide():
    """An indivisible axis is a refusal with a named reason, not a silent pass."""
    family, rung = parse_tessera_format_name("TESSERA_E2M1_K2_R896")
    ok, reason = tm.tessera_tp_legal(
        family, rung, (2048, 1000), tp_degree=8,
        parallel_kind=tm.PARALLEL_ROW)
    assert not ok
    assert reason


def test_tp_gate_reason_and_provenance_reach_the_candidate_gate():
    """The allocator's own applicability gate consumes the TP verdict."""
    from prismaquant import allocator_candidates as ac
    from prismaquant import format_registry as fr

    verdict = ac._tensor_parallel_applicability(
        "TESSERA_E2M1_K2_R896", qname="model.layers.0.self_attn.q_proj",
        target_profile="tessera_research_sm121",
        in_features=1024, out_features=2048, packed_expert=False,
    )
    assert verdict.provenance is not None
    assert "tp_degree" in verdict.provenance
    assert "tp_parallel_kind" in verdict.provenance


def test_tp_gate_is_inert_for_stock_formats():
    from prismaquant import allocator_candidates as ac
    from prismaquant import format_registry as fr

    verdict = ac._tensor_parallel_applicability(
        "NVFP4", qname="model.layers.0.mlp.down_proj",
        target_profile="tessera_research_sm121",
        in_features=3072, out_features=1024, packed_expert=False,
    )
    assert verdict.legal and verdict.provenance is None


# ---------------------------------------------------------------------------
# Gate 3: the route, attested or explicitly not
# ---------------------------------------------------------------------------

def test_route_admission_never_asserts():
    """Every admission names the table that answered (principle 14)."""
    adm = tm.route_admission("TESSERA_E2M1_K2_R896")
    assert adm.source
    assert adm.route_status in {
        tm.ROUTE_STATUS_BACKED, tm.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
        tm.ROUTE_STATUS_UNATTESTED,
    }


def test_attested_menu_is_closed_with_no_tessera_contract_pinned():
    """A measured fact about the pinned release, recorded as a test.

    The pinned Gridbook contract publishes no Tessera cell, and no Tessera
    RELEASE tag exists, so with the dev pin unset the DEFAULT menu holds no
    Tessera rung at all. This is the honest state of the gate, not a bug:
    principle 9 says a format is production-eligible only when a pinned
    runtime attests it. A RELEASE tag is what flips this in production; the
    dev pin below is what flips it here.
    """
    assert not trc.dev_pin_requested(), "the dev pin must not leak into this test"
    rungs = tm.expand_tessera_menu(SHAPE, mode=tm.MENU_ATTESTED)
    assert rungs == []


def test_the_dev_pin_attests_exactly_the_rungs_the_contract_publishes(dev_pin):
    """The attested menu is the contract's own cells, and it has no rate axis.

    This is the headline the pin buys and the honest limit of "allocate
    continuously": Tessera's packaged contract attests ONE rung per family, so
    the attested menu is a handful of points rather than a range, and the
    continuous axis is reachable only under the research menu.  Widening it is
    a change to the contract's ``attested_rungs_q256``, not to PrismaQuant.

    **Derived, not typed.**  This test asserted the literal two-element list
    ``[E2M1_K2_R896, E4M3_K1_R1024]`` and went red the day the runtime attested
    a third family -- reporting a *correct* menu as a defect, which is the one
    thing an anti-staleness test must never do.  What is pinned now is the
    rule: the attested menu is exactly the rungs the contract's own native
    cells cover, in the menu's order, one per family, and every one of them
    carries the cell's status rather than a status this file typed.
    """
    contract = trc.load_tessera_contract()
    expected = {
        f"{family}_R{rung}"
        for family in contract.reader_rate_range
        for rung in sorted(contract.attested_rungs.get(family, ()))
        if contract.native_cells(family, rung)
    }
    rungs = tm.expand_tessera_menu(SHAPE, mode=tm.MENU_ATTESTED)
    names = [r.format_name for r in rungs]
    assert set(names) == expected, (names, sorted(expected))
    assert len(names) == len(set(names)), names

    # No rate axis: at most one attested rung per family, so no family offers
    # a choice of rate on the default path.  This is the claim the docstring
    # makes, and it is the one that would quietly stop being true.
    families = [r.admission.payload_family for r in rungs]
    assert len(families) == len(set(families)), families

    for rung in rungs:
        # Read off the cell, not typed: these cells are backed_with_serve_flag.
        assert rung.admission.route_status == tm.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
        assert rung.admission.requires_serve_flags == (
            "TESSERA_SERVE_MODE=resident|streamed",)
        assert rung.admission.source.startswith("tessera_dev_pin:runtime_contract:")
        assert rung.admission.max_world_size == 1

    # ...and the derivation is not vacuous: the contract must actually attest
    # something, or an empty menu would pass every assertion above.
    assert expected, "the pinned contract attests no rung at all"


def test_a_rate_the_contract_does_not_publish_is_unattested_not_backed(dev_pin):
    """One q256 step off the published rung is absence of a claim."""
    on = tm.route_admission("TESSERA_E4M3_K1_R1024")
    off = tm.route_admission("TESSERA_E4M3_K1_R1023")
    assert on.route_status == tm.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    assert off.route_status == tm.ROUTE_STATUS_UNATTESTED
    assert "R1023" in off.detail and "[1024]" in off.detail


def test_a_prose_only_tessera_edit_does_not_re_stale_the_pin(dev_pin, monkeypatch, tmp_path):
    """Issue #38, the whole point: the pin gates on the ANSWER, not the bytes.

    Rewrite every prose field the contract carries, reorder its keys, bump
    ``contract_version``, and hand the reader a file with a different sha256.
    Not one value a gate reads has moved, so the read must succeed -- because
    principle 14 says prose explains and is never a value a gate reads, and a
    pin that fired on it turned PrismaQuant's attested path off on every
    Tessera commit that touched a comment.
    """
    import json as _json

    payload = _json.loads(trc.contract_path().read_text(encoding="utf-8"))
    payload["contract_version"] = int(payload["contract_version"]) + 1
    payload["changelog"] = [{"contract_version": 999, "change": "prose only"}]
    for entry in payload["formats"]:
        entry["detail"] = "rewritten prose that no gate reads"
    for cell in payload["lane_eligibility"]["cells"]:
        cell["rationale"] = "rewritten prose that no gate reads"
        cell["detail"] = "also rewritten"
    for unit in payload["tensor_parallel"].get("units", ()):
        for axis in unit.get("loader_axes", ()) or ():
            if isinstance(axis, dict):
                axis["reason"] = "rewritten prose that no gate reads"
    # Reverse every mapping's key order and both arrays the reader iterates.
    payload = {k: payload[k] for k in reversed(list(payload))}
    payload["formats"] = list(reversed(payload["formats"]))
    payload["lane_eligibility"]["cells"] = list(
        reversed(payload["lane_eligibility"]["cells"]))

    moved = tmp_path / "runtime_contract.json"
    moved.write_text(_json.dumps(payload, indent=1), encoding="utf-8")
    assert (hashlib.sha256(moved.read_bytes()).hexdigest()
            != trc.TESSERA_DEV_PIN_CONTRACT_SHA256), "the bytes must differ"
    monkeypatch.setattr(trc, "contract_path", lambda: moved)

    contract = trc.load_tessera_contract()
    assert contract is not None
    assert contract.contract_version == payload["contract_version"]
    ident = contract.identity()
    assert ident["bytes_are_the_reviewed_bytes"] is False, (
        "provenance must say the bytes drifted even though the answer did not")


def test_the_line_a_human_reads_says_the_bytes_are_not_the_reviewed_ones():
    """The drift flag has to appear where a person will see it.

    ``identity()`` has recorded ``bytes_are_the_reviewed_bytes`` since the pin
    became an answer pin, and it travels into ``layer_config.json``.  But the
    allocator's one human-facing line printed a contract sha with nothing to
    compare it against, so a run reading unreviewed bytes looked exactly like a
    run reading reviewed ones.  A field that exists only in a JSON blob nobody
    opens is a confession log, not a notice.
    """

    reviewed = dict(
        commit="c6d52e2b53e0fb4593e4fb828fab0f681c43563e",
        contract_sha256="b" * 64,
        reviewed_contract_sha256="b" * 64,
        bytes_are_the_reviewed_bytes=True,
        plugin_version="0.1.0",
        contract_version=7,
    )
    drifted = dict(reviewed,
                   contract_sha256="a" * 64,
                   reviewed_contract_sha256="b" * 64,
                   bytes_are_the_reviewed_bytes=False)

    quiet = trc.describe_dev_pin(reviewed)
    loud = trc.describe_dev_pin(drifted)

    # The matching case stays quiet: nothing to compare, nothing to say.
    assert "reviewed" not in quiet, quiet
    assert "bbbbbbbbbbbb" in quiet and "c6d52e2b53e0" in quiet

    # The drifted case names both shas and says the answer still matches, which
    # is why the run was allowed to proceed at all.
    assert "aaaaaaaaaaaa" in loud, loud
    assert "reviewed bbbbbbbbbbbb" in loud, loud
    assert "answer equal" in loud, loud

    # Both carry the identity fields a reader needs to reproduce the read.
    for line in (quiet, loud):
        assert "plugin=0.1.0" in line and "contract_v7" in line, line


def test_a_moved_answer_refuses_and_names_the_field(dev_pin, monkeypatch, tmp_path):
    """The other half: a value a gate reads moves, and the read raises.

    It must name the field. A refusal that only says "the contract changed"
    is the corruption warning #38 was filed about; the operator's next move is
    to review what moved, so the message has to carry it.
    """
    import json as _json

    payload = _json.loads(trc.contract_path().read_text(encoding="utf-8"))
    for entry in payload["formats"]:
        if entry["family"] == "TESSERA_E4M3_K1":
            entry["attested_rungs_q256"] = [1024, 1280]
            entry.pop("candidate_rungs_q256", None)
    moved = tmp_path / "runtime_contract.json"
    moved.write_text(_json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(trc, "contract_path", lambda: moved)

    with pytest.raises(trc.TesseraContractError) as exc:
        trc.load_tessera_contract()
    msg = str(exc.value)
    assert "re-review" in msg
    assert "families[TESSERA_E4M3_K1].attested_rungs_q256" in msg
    assert "[1024]" in msg and "[1024, 1280]" in msg
    assert "not a corruption warning" in msg


def test_a_new_cell_is_a_moved_answer_even_though_nothing_was_removed(
        dev_pin, monkeypatch, tmp_path):
    """Additive is not free. A cell PrismaQuant never reviewed attests routes.

    Tessera's changelog calls a family-adding edit ADDITIVE, and for a v1
    *reader* it is. For an *admission gate* it is not: a new cell admits units
    that no human on this side looked at. Contract v5 adding the BF16 family
    is exactly that edit, which is why the gate has to fire on it.
    """
    import copy
    import json as _json

    payload = _json.loads(trc.contract_path().read_text(encoding="utf-8"))
    cells = payload["lane_eligibility"]["cells"]
    extra = copy.deepcopy(cells[0])
    extra["id"] = extra["id"] + "_invented"
    cells.append(extra)
    moved = tmp_path / "runtime_contract.json"
    moved.write_text(_json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(trc, "contract_path", lambda: moved)

    with pytest.raises(trc.TesseraContractError, match="NEW, not in the reviewed"):
        trc.load_tessera_contract()


def test_the_reviewed_answer_is_the_installed_one(dev_pin):
    """Anti-staleness: the literal must describe the Tessera on this box.

    Without this the pin drifts the other way -- a literal nobody regenerated
    keeps passing against a contract nobody read, because the only thing that
    reads the literal is the gate that compares it to itself.
    """
    contract = trc.load_tessera_contract()
    assert trc.contract_answer(contract) == trc.TESSERA_DEV_PIN_ANSWER


def test_the_answer_excludes_every_field_a_gate_does_not_read(dev_pin):
    """``contract_answer`` is the principle-14 line, so state where it sits.

    Identity travels into provenance and must NOT be in the answer, or a
    version bump is a re-review; the values an admission decision is made of
    MUST be, or the gate is decorative.
    """
    answer = trc.contract_answer(trc.load_tessera_contract())
    assert set(answer) == {"schema", "lane_schema", "quant_method",
                           "families", "cells"}
    flat = repr(answer)
    for identity_field in ("contract_version", "plugin_version", "attested_on",
                           "rationale", "detail", "changelog"):
        assert identity_field not in flat, (
            f"{identity_field} is identity or prose, not an answer")
    for family in ("TESSERA_E2M1_K2", "TESSERA_E4M3_K1", "TESSERA_BF16_K1"):
        assert set(answer["families"][family]) == {
            "reader_rate_range_q256", "attested_rungs_q256", "max_world_size"}


def test_reading_the_contract_does_not_import_the_serving_plugin():
    """Path arithmetic, not ``importlib.resources.files('tessera.serving')``.

    Importing that package registers the vLLM plugin; a producer-side contract
    read must not have that side effect, and this task's brief forbids the
    import outright.
    """
    import subprocess
    import sys

    script = (
        "import sys\n"
        "from prismaquant import tessera_runtime_contract as trc\n"
        "import os\n"
        f"os.environ['{trc.TESSERA_DEV_PIN_ENV}'] = '{trc.TESSERA_DEV_PIN_COMMIT}'\n"
        "c = trc.load_tessera_contract()\n"
        "assert c is not None\n"
        "assert 'tessera.serving' not in sys.modules, sorted(\n"
        "    m for m in sys.modules if m.startswith('tessera.'))\n"
        "print('OK')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OK" in out.stdout


def test_tp_above_the_attested_world_size_is_refused_in_the_attested_menu(dev_pin):
    """Two legs, and the attestation one binds first.

    The contract's ``tensor_parallel`` block is ``closed_world`` and lists both
    families at ``max_world_size: 1``. A ``[2048, 1024]`` unit shards perfectly
    at TP=2 on either axis, so geometry alone would admit it; the runtime says
    it serves one rank, and that is the answer. The research menu is unmoved --
    it prices unattested rungs on purpose.
    """
    geometry_ok, _ = tm.tessera_tp_legal(
        "TESSERA_E4M3_K1", 1024, SHAPE,
        tp_degree=2, parallel_kind=tm.PARALLEL_COLUMN)
    assert geometry_ok, "the shape shards; only the attestation should refuse"
    legal, reason = tm.tessera_tp_legal(
        "TESSERA_E4M3_K1", 1024, SHAPE,
        tp_degree=2, parallel_kind=tm.PARALLEL_COLUMN,
        require_attested_world=True)
    assert not legal
    assert "unattested" in reason and "world size 1" in reason
    assert tm.expand_tessera_menu(
        SHAPE, mode=tm.MENU_ATTESTED, tp_degree=2,
        parallel_kind=tm.PARALLEL_COLUMN) == []
    assert tm.expand_tessera_menu(
        SHAPE, mode=tm.MENU_RESEARCH, tp_degree=2,
        parallel_kind=tm.PARALLEL_COLUMN)


def test_research_menu_is_dense_and_stamps_its_status():
    rungs = tm.expand_tessera_menu(SHAPE, mode=tm.MENU_RESEARCH)
    assert len(rungs) > 500, len(rungs)
    rates = {r.bits_per_param for r in rungs}
    assert len(rates) > 100, len(rates)
    assert all(r.admission.route_status == tm.ROUTE_STATUS_UNATTESTED
               for r in rungs)


def test_menu_mode_refuses_an_unknown_spelling():
    with pytest.raises(tm.TesseraMenuError):
        tm.menu_mode("sortof")


def test_bytes_are_exact_and_monotone_in_rate():
    from prismaquant import format_registry as fr

    rungs = sorted(
        tm.expand_tessera_menu(SHAPE, mode=tm.MENU_RESEARCH),
        key=lambda r: r.body_rate_q256,
    )
    by_family: dict[str, list] = {}
    for rung in rungs:
        by_family.setdefault(rung.family, []).append(rung)
    for family, group in by_family.items():
        prev = -1
        for rung in group:
            spec = fr.get_format(rung.format_name)
            assert spec.memory_bytes_for_shape(SHAPE) == rung.memory_bytes
            assert rung.memory_bytes >= prev, (family, rung.format_name)
            prev = rung.memory_bytes


# ---------------------------------------------------------------------------
# The reductions: what may be dropped, and what may never be
# ---------------------------------------------------------------------------

def test_prune_dominated_keeps_a_point_inside_the_hull():
    """The reduction is exact dominance. A convex hull is NOT admissible.

    (100, 10) and (300, 1) are the hull; (200, 6) sits strictly above the
    segment between them and a hull prune would drop it. The knapsack's budget
    is discrete, so at a remaining capacity that fits 200 bytes and not 300 it
    is the optimum -- dropping it would change the DP's answer.
    """
    rows = [(100, 10.0, "a"), (200, 6.0, "b"), (300, 1.0, "c")]
    kept = [r[2] for r in tm.prune_dominated(rows)]
    assert kept == ["a", "b", "c"]


def test_prune_dominated_drops_only_both_axes_worse():
    rows = [(100, 10.0, "a"), (150, 12.0, "worse-both"), (300, 1.0, "c")]
    kept = [r[2] for r in tm.prune_dominated(rows)]
    assert kept == ["a", "c"]


def test_collapse_to_dp_bins_agrees_with_the_solver():
    from prismaquant.allocator_solver import _charged_bins

    n_params, total = 2_048 * 1_024, 500_000_000
    rows = [
        (int(bpp * n_params / 8), 1.0 / bpp, f"r{i}")
        for i, bpp in enumerate([3.0, 3.001, 3.5, 4.0])
    ]
    kept = tm.collapse_to_dp_bins(
        rows, baseline_bits_per_param=3.0, n_params=n_params,
        total_params=total, bit_precision=1e-4)
    bins = {
        _charged_bins(
            (row[0] * 8.0 / n_params - 3.0) * (n_params / total), 1e-4)
        for row in kept
    }
    assert len(bins) == len(kept)


def test_reduce_continuous_menu_is_a_noop_without_tessera():
    from prismaquant.allocator_candidates import Candidate, reduce_continuous_menu

    cands = {
        "x": [
            Candidate(fmt="NVFP4", bits_per_param=4.5, memory_bytes=100,
                      predicted_dloss=1.0),
            Candidate(fmt="FP8_E4M3", bits_per_param=8.0, memory_bytes=200,
                      predicted_dloss=0.5),
        ]
    }
    report: dict = {}
    out = reduce_continuous_menu(
        cands, {"x": {"n_params": 1000}}, bit_precision=1e-4, report=report)
    assert out == cands
    assert report == {}


# ---------------------------------------------------------------------------
# Promotion: the unit shares a family, not a rate
# ---------------------------------------------------------------------------

def test_promotion_class_is_the_family_for_tessera_and_the_name_otherwise():
    assert format_promotion_class("TESSERA_E2M1_K2_R896") == "TESSERA_E2M1_K2"
    assert format_promotion_class("NVFP4") == "NVFP4"
    assert format_promotion_class("FP8_E4M3") == "FP8_E4M3"


def test_fused_siblings_share_a_family_with_rates_free():
    from prismaquant.allocator_solver import _promote_group_components

    rank = {
        "TESSERA_E2M1_K2_R512": 0,
        "TESSERA_E2M1_K2_R896": 1,
        "TESSERA_E4M3_K1_R896": 2,
    }
    legal = {name: set(rank) for name in ("q", "k", "v")}
    out = _promote_group_components(
        {"q": "TESSERA_E2M1_K2_R512",
         "k": "TESSERA_E2M1_K2_R896",
         "v": "TESSERA_E2M1_K2_R512"},
        rank, [["q", "k", "v"]], legal,
    )
    assert {format_promotion_class(f) for f in out.values()} == {
        "TESSERA_E2M1_K2"}
    # ...and the rates did NOT collapse to one.
    assert len(set(out.values())) == 2, out


def test_promotion_is_non_degrading_within_the_family():
    from prismaquant.allocator_solver import _promote_group_components

    rank = {"TESSERA_E2M1_K2_R512": 0, "TESSERA_E2M1_K2_R896": 1}
    legal = {"q": set(rank), "k": {"TESSERA_E2M1_K2_R896"}}
    out = _promote_group_components(
        {"q": "TESSERA_E2M1_K2_R512", "k": "TESSERA_E2M1_K2_R896"},
        rank, [["q", "k"]], legal)
    assert out["k"] == "TESSERA_E2M1_K2_R896"
    assert out["q"] == "TESSERA_E2M1_K2_R512"


def test_promotion_unchanged_for_a_stock_menu():
    from prismaquant.allocator_solver import _promote_group_components

    rank = {"NVFP4": 0, "FP8_E4M3": 1, "BF16": 2}
    legal = {n: set(rank) for n in ("q", "k", "v")}
    out = _promote_group_components(
        {"q": "NVFP4", "k": "FP8_E4M3", "v": "NVFP4"},
        rank, [["q", "k", "v"]], legal)
    assert out == {"q": "FP8_E4M3", "k": "FP8_E4M3", "v": "FP8_E4M3"}


def test_promotion_falls_back_to_uniform_when_the_family_is_illegal():
    """The relaxation may only widen what promotion accepts, never fail."""
    from prismaquant.allocator_solver import _promote_group_components

    rank = {"TESSERA_E2M1_K2_R896": 0, "BF16": 1}
    legal = {"q": {"TESSERA_E2M1_K2_R896", "BF16"}, "k": {"BF16"}}
    out = _promote_group_components(
        {"q": "TESSERA_E2M1_K2_R896", "k": "BF16"},
        rank, [["q", "k"]], legal)
    assert out == {"q": "BF16", "k": "BF16"}


# ---------------------------------------------------------------------------
# The route travels with the choice
# ---------------------------------------------------------------------------

def test_serving_lane_route_resolves_a_tessera_rung():
    """A Tessera rung reports 'unattested', never 'no_declared_lane'.

    Absence of a declaration and a declaration of absence are different facts,
    and principle 9 wants the second one countable.
    """
    from prismaquant.serving_profiles import serving_lane_route

    lane = serving_lane_route(
        "tessera_research_sm121", "TESSERA_E2M1_K2_R896")
    assert lane is not None
    assert lane.route_status == tm.ROUTE_STATUS_UNATTESTED
    assert lane.route_status_source
    assert lane.activation_contract


def test_selection_provenance_counts_tessera_units_by_route_status():
    from prismaquant.allocator_candidates import selection_serving_lane_provenance

    prov = selection_serving_lane_provenance(
        {"a": "TESSERA_E2M1_K2_R896", "b": "TESSERA_E4M3_K1_R896"},
        None, "tessera_research_sm121")
    assert prov["units_without_declared_lane"] == 0
    assert prov["route_status_counts"].get(tm.ROUTE_STATUS_UNATTESTED) == 2


def test_layer_config_roundtrip_recovers_the_rung():
    from prismaquant.layer_config import canonicalize_format
    from prismaquant import format_registry as fr

    name = "TESSERA_E2M1_K2_R896"
    entry = fr.get_format(name).autoround_config()
    assert entry["data_type"] == "tessera"
    assert canonicalize_format(entry) == name


# ---------------------------------------------------------------------------
# Where the relaxation is, and is not, reached
# ---------------------------------------------------------------------------

def test_pre_aggregation_offers_a_fused_group_one_family_and_free_rates():
    """The DP sees the group's exact knapsack, not one shared rung.

    This test used to pin the opposite, and said a later change that made the
    DP mixed-rung capable would have to come here and say so. This is that
    change, and here is what it says.

    The old aggregation built one super-item candidate per format **name**, by
    intersecting the members' menus, so the DP could only return a single rung
    for a whole fused group. Two things were wrong with reading that as the
    serving constraint. It is too tight: the runtime dispatches on the
    **decoder** -- the grid and arity a Tessera family names -- and a shared
    rate is not a dispatch property, while q_proj and k_proj are different
    tensors with different sensitivities. And on a continuous rate axis it is
    ruinous rather than merely conservative: the intersection of two members'
    *measured* menus can be a single rung, which is how a group's whole menu
    collapses for a reason that has nothing to do with Tessera.

    So the group's option set is now the Minkowski sum of its members' (bytes,
    cost) menus restricted to one family -- the group's own exact multi-choice
    knapsack, kept as a Pareto set under dominance (never a hull). Every option
    is one family across the group and a rate per member.

    **The serving premise is still unattested.** Whether a runtime decodes a
    fused group as per-member wires it concatenates -- which is what free
    per-member rates require, since ``bresenham_rate_schedule(root,
    n_columns)`` is a per-COLUMN quota shared by every row of ONE unit -- is a
    fact about that runtime, and no pinned release attests it. Principle 9's
    export gate is what decides whether such an assignment ships; this pins
    only what the allocator may consider.
    """
    from prismaquant.allocator_candidates import tessera_group_composites
    from prismaquant.allocator_solver import Candidate

    members = ["m.q_proj", "m.k_proj"]
    candidates = {
        members[0]: [
            Candidate(fmt="TESSERA_E4M3_K1_R100", bits_per_param=0.0,
                      memory_bytes=10, predicted_dloss=9.0),
            Candidate(fmt="TESSERA_E4M3_K1_R200", bits_per_param=0.0,
                      memory_bytes=20, predicted_dloss=1.0),
        ],
        members[1]: [
            Candidate(fmt="TESSERA_E4M3_K1_R100", bits_per_param=0.0,
                      memory_bytes=10, predicted_dloss=5.0),
            Candidate(fmt="TESSERA_E4M3_K1_R300", bits_per_param=0.0,
                      memory_bytes=20, predicted_dloss=4.5),
        ],
    }
    options = tessera_group_composites(members, candidates, n_params=1000)
    # The members share exactly one rung name; the old intersection menu would
    # have offered one option. The knapsack offers the whole frontier.
    assert len({c.fmt for c in candidates[members[0]]}
               & {c.fmt for c in candidates[members[1]]}) == 1
    assert len(options) > 1
    mixed = [o for o in options if len(set(o.member_formats.values())) > 1]
    assert mixed, "no mixed-rung option was offered"
    for option in options:
        families = {f.rsplit("_R", 1)[0] for f in option.member_formats.values()}
        assert families == {"TESSERA_E4M3_K1"}, families


def test_the_reduction_is_reapplied_after_aggregation():
    """The super item's menu passes through the same two exact reductions.

    Super items are built from ``specs_sorted`` directly, bypassing the
    reduction ``build_candidates`` applies per Linear. Without a second pass a
    fused group would carry the full unreduced Tessera axis into the DP.
    """
    import inspect
    from prismaquant import allocator

    src = inspect.getsource(allocator)
    head = src.split("post_aggregation_availability")[0]
    assert "reduce_continuous_menu(" in head.split(
        "aggregate_fused_siblings(")[-1]


# ---------------------------------------------------------------------------
# Gate 12: the TESSERA menu token narrows to what the runtime attests
#
# A cost table is priced under whatever menu mode the CAMPAIGN ran; the
# attested set is a property of the RUNTIME.  A research-priced table read back
# on the default path therefore holds thousands of columns the pinned contract
# does not publish, and the question is what the token does with them.  Before
# this gate it expanded to all of them and ``require_producer_formats`` refused
# the whole run -- so the default path could not allocate at all, and the
# backed rungs never reached the DP.  These three tests pin the split: the
# TOKEN narrows and says so, an explicit NAME still refuses.
# ---------------------------------------------------------------------------

PRICED = [
    "TESSERA_E2M1_K2_R896",     # attested
    "TESSERA_E2M1_K2_R640",     # serialisable, unattested rate
    "TESSERA_E4M3_K1_R1024",    # attested
    "TESSERA_E4M3_K1_R512",     # serialisable, unattested rate
    "TESSERA_E2M1_K1_R256",     # family the contract does not publish
]


def test_the_menu_token_expands_to_the_attested_subset_and_reports_the_rest(dev_pin):
    """The default path allocates over the backed axis, not over nothing."""
    menu, dropped = tm.expand_menu_tokens_report(
        ["NVFP4", tm.MENU_TOKEN, "BF16"], PRICED)
    assert menu == [
        "NVFP4", "TESSERA_E2M1_K2_R896", "TESSERA_E4M3_K1_R1024", "BF16",
    ], menu
    # the narrowing is reported, not silent: an allocation over 2 rungs and one
    # over 2423 must not look the same in a log (P12).
    assert sorted(dropped) == sorted([
        "TESSERA_E2M1_K2_R640", "TESSERA_E4M3_K1_R512", "TESSERA_E2M1_K1_R256",
    ]), dropped
    # and the filter is the same predicate the guard refuses on, so the token
    # can never expand to something ``require_producer_formats`` then rejects.
    fr.require_producer_formats(menu, where="test")


def test_an_explicitly_named_unattested_rung_still_refuses(dev_pin):
    """Only the token narrows. A human naming a reader-only rung is an error."""
    menu, dropped = tm.expand_menu_tokens_report(
        ["TESSERA_E2M1_K2_R640"], PRICED)
    assert menu == ["TESSERA_E2M1_K2_R640"]
    assert dropped == [], "no token, no narrowing report"
    with pytest.raises(ValueError, match="producer-eligible"):
        fr.require_producer_formats(menu, where="test")


def test_the_research_menu_token_keeps_every_priced_rung(monkeypatch, dev_pin):
    """Research prices the whole realisable axis on purpose; nothing narrows."""
    monkeypatch.setenv(tm.MENU_MODE_ENV, tm.MENU_RESEARCH)
    menu, dropped = tm.expand_menu_tokens_report([tm.MENU_TOKEN], PRICED)
    assert menu == PRICED, menu
    assert dropped == []


# ---------------------------------------------------------------------------
# Gate 13: the surrogate does not rank Tessera rungs, and the artifact says so
#
# Measured 2026-09-02, served: this menu's own allocations lose 2.00x in
# KL-vs-BF16 to a byte-matched uniform arm at 4.0 bpp, and the loss is in the
# cost model on the units it priced. The repair is not this seam's; recording
# it where a MACHINE can read it is. A warning printed to a terminal is not a
# property of the artifact -- these two tests are what keep it one.
# ---------------------------------------------------------------------------

def test_the_selection_caveat_records_the_served_measurement():
    caveat = tm.surrogate_selection_caveat()
    assert caveat["surrogate_ranks_tessera_rungs"] == "measured_false"
    assert caveat["measurement"].endswith(
        "tessera-allocated-served-2026-09-02.md")
    # the three served ratios, not a prose summary a gate cannot read (P14)
    assert caveat["served_kl_allocated_over_byte_matched_uniform"] == {
        "3.0": 2.33, "4.0": 2.00, "5.0": 2.88,
    }
    assert caveat["priced_units_only"]["ratio"] == 1.93
    # both requirements, because the first alone is measured insufficient:
    # the validated frontier re-scores only the allocator's own Pareto points.
    requires = caveat["requires_before_promotion"]
    assert any("validated-surrogate" in r for r in requires), requires
    assert any("uniform" in r for r in requires), requires


def test_the_allocator_stamps_the_caveat_rather_than_only_printing_it():
    """A terminal line is not a property of the artifact."""
    import inspect
    from prismaquant import allocator

    src = inspect.getsource(allocator)
    assert src.count("surrogate_selection_caveat()") >= 2, (
        "the caveat must reach BOTH provenance stamp sites, not just the "
        "layer_config one")
    assert "CANDIDATE, not a selection" in src


# ---------------------------------------------------------------------------
# The menu cache bound (tessera#46)
#
# These pin the RULE -- "a memo must survive one shape's widest menu" -- and
# never the roster.  A number written here would be the same mistake the bug
# was: ``maxsize=4096`` was correct arithmetic against a three-family menu and
# silently wrong the moment a fourth family was admitted.
# ---------------------------------------------------------------------------

def _menu_memos():
    from prismaquant import tessera_footprint as tf

    return {
        "tessera_menu._shard_geometry": tm._shard_geometry,
        "tessera_footprint._exact_bits_for_shape": tf._exact_bits_for_shape,
    }


def test_the_menu_rung_count_is_asked_of_the_families_not_listed():
    """The first factor of the bound is computed, so a new family moves it."""
    expected = sum(
        spec.mathematical_q256_bounds[1] - spec.mathematical_q256_bounds[0] + 1
        for spec in tm.menu_families()
    )
    assert tm.menu_rungs_per_shape() == expected
    # and it really is the ceiling on what one pass over one shape fills --
    # today also the exact count, since every rung is legal at this shape, but
    # the rule is the ceiling.  A family whose span or Bresenham root refuses
    # some shapes would lower the fill without touching the bound, and pinning
    # equality here would fail for a reason that has nothing to do with the
    # cache.
    for memo in _menu_memos().values():
        memo.cache_clear()
    tm.expand_tessera_menu((256, 256), mode=tm.MENU_RESEARCH, step_q256=1)
    from prismaquant import tessera_footprint as tf

    fills = tf._exact_bits_for_shape.cache_info().misses
    assert 0 < fills <= tm.menu_rungs_per_shape()


def test_every_menu_memo_survives_one_shapes_widest_menu(monkeypatch):
    """The rule.  Not ``maxsize == <number>`` -- ``maxsize >= one shape``.

    A memo smaller than the widest menu one shape can produce evicts its own
    entries while that shape is still being priced, which is tessera#46
    exactly: 6764 rungs into a 4096-entry memo, cold 10.06 s and warm 9.958 s.
    """
    floor = tm.menu_rungs_per_shape()
    for name, memo in _menu_memos().items():
        assert memo.cache_info().maxsize >= floor, (
            f"{name} holds {memo.cache_info().maxsize} entries against a "
            f"{floor}-rung menu: one shape evicts itself")

    # By construction, not by the value of a constant: the bound is a whole
    # number of shapes and at least one, so no setting of the knob can put it
    # under a shape.
    for shapes in ("1", "3", "25", "512"):
        monkeypatch.setenv(tm.MENU_CACHE_SHAPES_ENV, shapes)
        assert tm.menu_cache_bound() == floor * int(shapes)
        assert tm.menu_cache_bound() >= floor
    for refused in ("0", "-4", "", "  ", "lots", "1.5"):
        monkeypatch.setenv(tm.MENU_CACHE_SHAPES_ENV, refused)
        if refused.strip() == "":
            assert tm.menu_cache_shapes() == tm.DEFAULT_MENU_CACHE_SHAPES
            continue
        with pytest.raises(tm.TesseraMenuError):
            tm.menu_cache_shapes()


def test_a_second_pass_over_one_shape_recomputes_nothing():
    """The behaviour the bound buys, measured on the memo rather than a clock.

    A wall-clock assertion would be a coin flip on a loaded box; hits and
    misses say the same thing and say it exactly.
    """
    from prismaquant import tessera_footprint as tf

    for memo in _menu_memos().values():
        memo.cache_clear()
    shape = (256, 256)
    first = tm.expand_tessera_menu(shape, mode=tm.MENU_RESEARCH, step_q256=1)
    cold = tf._exact_bits_for_shape.cache_info()
    assert cold.hits == 0 and cold.misses > 0
    second = tm.expand_tessera_menu(shape, mode=tm.MENU_RESEARCH, step_q256=1)
    warm = tf._exact_bits_for_shape.cache_info()
    assert first == second
    assert warm.misses == cold.misses, (
        "a repeat pass over one shape recomputed rungs it had already priced")
    assert warm.hits == cold.misses


def test_the_geometry_memo_is_keyed_by_family_name_not_by_the_family():
    """One answer, one key.  A spec and its name would be two.

    ``get_tessera_family`` takes either, so the two spellings agree on the
    answer and disagree on the key -- which halves the hit rate against a bound
    that was sized in rungs, without changing a single number.
    """
    spec = tm.menu_families()[0]
    lo, _hi = spec.mathematical_q256_bounds
    tm._shard_geometry.cache_clear()
    by_name = tm._shard_geometry(spec.name, lo, 2048, 1024)
    after_one = tm._shard_geometry.cache_info()
    assert after_one.misses == 1 and after_one.currsize == 1
    assert tm._shard_geometry(spec.name, lo, 2048, 1024) is by_name
    assert tm._shard_geometry.cache_info().hits == 1
    with pytest.raises(tm.TesseraMenuError):
        tm._shard_geometry(spec, lo, 2048, 1024)
