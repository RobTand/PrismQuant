"""Issue #136: nothing reads Tessera's route `decoder`.

Tessera's runtime contract publishes, per native extension, what a serve
does when the `.so` cannot build: resident mode keeps serving on a NAMED
substitute decoder and stamps it on every route record (the field exists so
"a receipt must never claim the native decoder for a serve that took the"
fallback).  Nothing in PrismaQuant read it: a Tessera shipcard could price
`TESSERA_NVFP4` W4A4 while a serve produced every number on
`torch_materialize_stock` with nothing refusing.

The gate refuses a receipt whose served routes ran on a substitute decoder,
whose records lack a decoder, or whose served route set disagrees with the
priced one -- and the shipcard requires the receipt on every rate-axis
(Tessera) artifact, replayed from the carried records at publication.

Substitute names are DERIVED from the pinned contract's answer
(``tessera_runtime_contract.TESSERA_DEV_PIN_ANSWER``), never typed: a test
that hardcodes today's decoder would pass with a stale transcription, which
is the defect, not the check.  The stand-in native decoder below is
synthetic on purpose -- this repository does not publish the native
decoder's name, so the gate's rule is "refuse the derived substitute set
and anything decoder-less", and the receipt stamps what it observed.
"""
from __future__ import annotations

import json

import pytest

NATIVE = "native_test_decoder"


def _substitutes():
    from prismaquant import tessera_runtime_contract as trc

    answer = trc.TESSERA_DEV_PIN_ANSWER
    decoders = set()
    for row in answer["native_extensions"]:
        for behaviour in row["when_unavailable"].values():
            if behaviour["decoder"] is not None:
                decoders.add(behaviour["decoder"])
    assert decoders, "the pinned answer must name a substitute decoder"
    return sorted(decoders)


def _native_records(*routes):
    return [{"route": route, "decoder": NATIVE, "count": 128}
            for route in routes]


# --- the gate ----------------------------------------------------------------

def test_a_serve_on_the_substitute_decoder_is_refused():
    from prismaquant.tessera_route_receipt import check_route_receipt

    substitutes = _substitutes()
    verdict = check_route_receipt(
        priced_routes=["TESSERA_NVFP4"],
        route_records=[{"route": "TESSERA_NVFP4",
                        "decoder": substitutes[0], "count": 128}],
        substitute_decoders=substitutes,
    )
    assert verdict["passed"] is False
    assert substitutes[0] in verdict["detail"]
    assert verdict["substitute_hits"]


def test_a_record_without_a_decoder_is_refused_not_read_as_native():
    from prismaquant.tessera_route_receipt import (
        TesseraRouteReceiptError,
        check_route_receipt,
    )

    substitutes = _substitutes()
    for records in ([{"route": "TESSERA_NVFP4", "count": 128}],
                    [{"route": "TESSERA_NVFP4", "decoder": "",
                      "count": 128}]):
        with pytest.raises(TesseraRouteReceiptError, match="no decoder"):
            check_route_receipt(
                priced_routes=["TESSERA_NVFP4"],
                route_records=records,
                substitute_decoders=substitutes,
            )


def test_an_empty_census_is_refused():
    from prismaquant.tessera_route_receipt import check_route_receipt

    verdict = check_route_receipt(
        priced_routes=["TESSERA_NVFP4"],
        route_records=[],
        substitute_decoders=_substitutes(),
    )
    assert verdict["passed"] is False


def test_priced_vs_served_disagreement_is_refused_both_ways():
    from prismaquant.tessera_route_receipt import check_route_receipt

    substitutes = _substitutes()
    unserved = check_route_receipt(
        priced_routes=["TESSERA_NVFP4", "TESSERA_BF16_K1"],
        route_records=_native_records("TESSERA_NVFP4"),
        substitute_decoders=substitutes,
    )
    assert unserved["passed"] is False
    assert unserved["unserved_priced"] == ["TESSERA_BF16_K1"]

    unpriced = check_route_receipt(
        priced_routes=["TESSERA_NVFP4"],
        route_records=_native_records("TESSERA_NVFP4", "TESSERA_BF16_K1"),
        substitute_decoders=substitutes,
    )
    assert unpriced["passed"] is False
    assert unpriced["unpriced_served"] == ["TESSERA_BF16_K1"]


def test_a_native_serve_covering_the_priced_routes_passes_and_stamps_them():
    from prismaquant.tessera_route_receipt import check_route_receipt

    verdict = check_route_receipt(
        priced_routes=["TESSERA_NVFP4"],
        route_records=_native_records("TESSERA_NVFP4"),
        substitute_decoders=_substitutes(),
    )
    assert verdict["passed"] is True
    assert verdict["served_routes"] == ["TESSERA_NVFP4"]
    assert verdict["served_decoders"] == [NATIVE]


def test_a_gate_that_knows_no_substitute_is_not_a_gate():
    """With an empty substitute set every serve passes, so the constructor
    refuses rather than returning a verdict that detects nothing."""
    from prismaquant.tessera_route_receipt import (
        TesseraRouteReceiptError,
        check_route_receipt,
    )

    with pytest.raises(TesseraRouteReceiptError, match="substitute"):
        check_route_receipt(
            priced_routes=["TESSERA_NVFP4"],
            route_records=_native_records("TESSERA_NVFP4"),
            substitute_decoders=[],
        )


def test_malformed_records_are_refused_with_the_index():
    from prismaquant.tessera_route_receipt import parse_route_records

    with pytest.raises(ValueError, match="\\[1\\]"):
        parse_route_records([{"route": "TESSERA_NVFP4", "decoder": NATIVE},
                             {"route": "TESSERA_NVFP4"}])


# --- the shipcard slot ---------------------------------------------------------

def _tessera_card(tmp_path):
    from prismaquant.shipcard import build_shipcard, write_shipcard

    model_dir = tmp_path / "exported"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(
        {"model_type": "qwen3",
         "quantization_config": {"quant_method": "tessera"}}))
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    # A Tessera-lane card, as `lane_shipcard open --lane tessera` opens it:
    # the lane union -- not a rate-axis special case -- is what opens and
    # requires the route.census slot.
    card = build_shipcard(model_dir, lane="tessera")
    write_shipcard(model_dir / "shipcard.json", card)
    return model_dir


def _passing_census_record(model_dir):
    from prismaquant.shipcard import compute_model_sha, make_route_census_record

    return make_route_census_record(
        tool="test",
        model_sha=compute_model_sha(model_dir),
        priced_routes=["TESSERA_NVFP4"],
        route_records=_native_records("TESSERA_NVFP4"),
        substitute_decoders=_substitutes(),
    )


def test_a_tessera_artifact_owes_a_route_census_receipt(tmp_path):
    from prismaquant.shipcard import (
        ROUTE_CENSUS_SLOT,
        load_shipcard,
        required_slots,
        verify,
    )

    model_dir = _tessera_card(tmp_path)
    card = load_shipcard(model_dir / "shipcard.json")
    assert ROUTE_CENSUS_SLOT in required_slots(card, model_dir=model_dir)
    assert f"{ROUTE_CENSUS_SLOT}: UNFILLED" in verify(
        card, model_dir=model_dir)


def test_a_non_tessera_artifact_owes_no_route_census(tmp_path):
    from prismaquant.shipcard import (
        ROUTE_CENSUS_SLOT,
        build_shipcard,
        load_shipcard,
        required_slots,
        write_shipcard,
    )

    model_dir = tmp_path / "exported"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    card = build_shipcard(model_dir)
    write_shipcard(model_dir / "shipcard.json", card)
    assert ROUTE_CENSUS_SLOT not in required_slots(
        load_shipcard(model_dir / "shipcard.json"), model_dir=model_dir)


def test_a_tessera_card_without_a_lane_key_owes_no_census(tmp_path):
    """The requirement travels with the lane declaration, not with the
    quant_method sniff: a historical card carrying no `lane` verifies
    against exactly the base set it was opened with.  An earlier revision
    of this fix required the slot for every rate-axis artifact in
    `required_slots`; that second requirement path is dropped in favour of
    the lane union, which is the one mechanism a lane can add requirements
    through."""
    from prismaquant.shipcard import (
        ROUTE_CENSUS_SLOT,
        build_shipcard,
        load_shipcard,
        required_slots,
        write_shipcard,
    )

    model_dir = tmp_path / "exported"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(
        {"model_type": "qwen3",
         "quantization_config": {"quant_method": "tessera"}}))
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    card = build_shipcard(model_dir)
    write_shipcard(model_dir / "shipcard.json", card)
    assert "lane" not in load_shipcard(model_dir / "shipcard.json")
    assert ROUTE_CENSUS_SLOT not in required_slots(
        load_shipcard(model_dir / "shipcard.json"), model_dir=model_dir)


def test_a_passed_flag_on_substitute_records_does_not_verify(tmp_path):
    """The replay reads the carried records, not the carried boolean: a
    `passed=true` stamped over a substitute-decoder serve still refuses."""
    from prismaquant.shipcard import (
        ROUTE_CENSUS_SLOT,
        compute_model_sha,
        fill_slot,
        load_shipcard,
        make_route_census_record,
        verify,
    )

    model_dir = _tessera_card(tmp_path)
    substitutes = _substitutes()
    record = make_route_census_record(
        tool="test",
        model_sha=compute_model_sha(model_dir),
        priced_routes=["TESSERA_NVFP4"],
        route_records=[{"route": "TESSERA_NVFP4",
                        "decoder": substitutes[0], "count": 128}],
        substitute_decoders=substitutes,
    )
    record["passed"] = True
    path = model_dir / "shipcard.json"
    fill_slot(path, ROUTE_CENSUS_SLOT, record)
    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any(p.startswith(f"{ROUTE_CENSUS_SLOT}:") and "substitute" in p
               for p in problems), problems


def test_a_native_census_receipt_closes_the_slot(tmp_path):
    from prismaquant.shipcard import (
        ROUTE_CENSUS_SLOT,
        fill_slot,
        load_shipcard,
        verify,
    )

    model_dir = _tessera_card(tmp_path)
    path = model_dir / "shipcard.json"
    fill_slot(path, ROUTE_CENSUS_SLOT, _passing_census_record(model_dir))
    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert not [p for p in problems
                if p.startswith(f"{ROUTE_CENSUS_SLOT}:")], problems


def test_fill_route_census_cli_closes_the_slot_from_files(tmp_path):
    from prismaquant.shipcard import ROUTE_CENSUS_SLOT, load_shipcard, verify
    from prismaquant.shipcard_cli import main as shipcard_cli

    model_dir = _tessera_card(tmp_path)
    census = tmp_path / "census.json"
    census.write_text(json.dumps(_native_records("TESSERA_NVFP4")))
    assert shipcard_cli([
        "fill-route-census", str(model_dir / "shipcard.json"),
        "--census", str(census),
        "--priced-route", "TESSERA_NVFP4",
        "--substitute-decoder", _substitutes()[0],
        "--model-dir", str(model_dir),
    ]) == 0
    card = load_shipcard(model_dir / "shipcard.json")
    assert card["slots"][ROUTE_CENSUS_SLOT]["passed"] is True
    problems = verify(card, model_dir=model_dir)
    assert not [p for p in problems
                if p.startswith(f"{ROUTE_CENSUS_SLOT}:")], problems


def test_fill_route_census_cli_refuses_a_decoder_less_census(tmp_path):
    from prismaquant.shipcard import ROUTE_CENSUS_SLOT, load_shipcard
    from prismaquant.shipcard_cli import main as shipcard_cli

    model_dir = _tessera_card(tmp_path)
    census = tmp_path / "census.json"
    census.write_text(json.dumps([{"route": "TESSERA_NVFP4", "count": 3}]))
    assert shipcard_cli([
        "fill-route-census", str(model_dir / "shipcard.json"),
        "--census", str(census),
        "--priced-route", "TESSERA_NVFP4",
        "--substitute-decoder", _substitutes()[0],
        "--model-dir", str(model_dir),
    ]) == 2
    assert load_shipcard(
        model_dir / "shipcard.json")["slots"][ROUTE_CENSUS_SLOT] is None
