"""The measured-dispatch-table schema and its provenance refusal (P5c).

gridbook ``docs/audits/ultraplan_perf_2026-08-01.md`` §6 P5c asks the producer
to price serving from "the measured per-format × M-regime dispatch table from
§2". The load-time contract this file pins is what keeps that table honest:

1. **A row without a source is an error.** The whole value of the table is
   that the allocator can no longer discover a serving trade at the release
   gate; a fabricated row replaces one blind spot with a worse one, because
   the missing row is at least visible.
2. **One arena, one denominator.** Published serving numbers are ratios
   against different references (the 27B dense-prefill 1.44× is against a
   native artifact; the fused mid-M 1.04×/1.26×/1.45× are against FP8-CB's own
   expand+GEMM route). A row whose arena is undeclared is uninterpretable.
3. **Isolated-operator microbenchmarks are marked, not laundered.** Policy §5:
   "Raw standalone kernel timing is never served evidence."

The shipped example table is also pinned here: every row must cite a source,
and the table must declare itself proposal data.
"""
from __future__ import annotations

import copy
import json

import pytest

from prismaquant.serve_dispatch_table import (
    SCHEMA,
    DispatchTableError,
    dispatch_family_for_format,
    example_table_path,
    load_dispatch_table,
    missing_family_report,
    parse_dispatch_table,
)

_PROV = {
    "source": "tests/test_serve_dispatch_table.py synthetic fixture",
    "date": "2026-08-01",
    "gpu": "synthetic",
    "measured_quantity": "synthetic relative prefill cost",
    "units": "dimensionless",
    "derivation": "fixture constant",
}


def _table(**over):
    payload = {
        "schema": SCHEMA,
        "table_id": "fixture",
        "status": "proposal_data",
        "description": "fixture",
        "arenas": [
            {
                "phase": "prefill",
                "m_regime": "dense",
                "m": 1400,
                "reference_route": "native",
                "metric": "ttft_ms",
                "absolute_value": 1000.0,
                "statistic": "p95",
                "provenance": dict(_PROV),
            },
        ],
        "rows": [
            {
                "format_family": "NVFP4",
                "phase": "prefill",
                "m_regime": "dense",
                "lane": "native",
                "relative_unit_cost": 1.0,
                "provenance": dict(_PROV),
            },
        ],
    }
    payload.update(over)
    return payload


# ---------------------------------------------------------------------------
# 1. Provenance is mandatory: an uncited row is a fabricated measurement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field",
    ["source", "date", "gpu", "measured_quantity", "units", "derivation"],
)
def test_row_without_provenance_field_is_refused(field):
    payload = _table()
    del payload["rows"][0]["provenance"][field]
    with pytest.raises(DispatchTableError) as exc:
        parse_dispatch_table(payload)
    assert field in str(exc.value)
    assert "fabricated" in str(exc.value) or "uncited" in str(exc.value)


def test_row_with_blank_source_is_refused():
    """Blank is not "present": an empty source cites nothing."""
    payload = _table()
    payload["rows"][0]["provenance"]["source"] = "   "
    with pytest.raises(DispatchTableError, match="source"):
        parse_dispatch_table(payload)


def test_row_with_no_provenance_object_at_all_is_refused():
    payload = _table()
    del payload["rows"][0]["provenance"]
    with pytest.raises(DispatchTableError,
                       match="fabricated measurement|required field"):
        parse_dispatch_table(payload)


def test_arena_without_provenance_is_refused():
    payload = _table()
    del payload["arenas"][0]["provenance"]["gpu"]
    with pytest.raises(DispatchTableError, match="gpu"):
        parse_dispatch_table(payload)


# ---------------------------------------------------------------------------
# 2. Schema validation
# ---------------------------------------------------------------------------
def test_wrong_schema_is_refused():
    with pytest.raises(DispatchTableError, match="schema"):
        parse_dispatch_table(_table(schema="something.else.v9"))


def test_status_is_mandatory():
    payload = _table()
    payload["status"] = ""
    with pytest.raises(DispatchTableError, match="status"):
        parse_dispatch_table(payload)


def test_unknown_phase_is_refused():
    payload = _table()
    payload["rows"][0]["phase"] = "blended"
    with pytest.raises(DispatchTableError, match="phase"):
        parse_dispatch_table(payload)


def test_unknown_lane_is_refused():
    payload = _table()
    payload["rows"][0]["lane"] = "magic"
    with pytest.raises(DispatchTableError, match="lane"):
        parse_dispatch_table(payload)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_cost_is_refused(bad):
    payload = _table()
    payload["rows"][0]["relative_unit_cost"] = bad
    with pytest.raises(DispatchTableError, match="relative_unit_cost"):
        parse_dispatch_table(payload)


def test_row_naming_an_undeclared_arena_is_refused():
    payload = _table()
    payload["rows"][0]["m_regime"] = "not_declared"
    with pytest.raises(DispatchTableError, match="does not declare"):
        parse_dispatch_table(payload)


def test_duplicate_arena_is_refused():
    payload = _table()
    payload["arenas"].append(copy.deepcopy(payload["arenas"][0]))
    with pytest.raises(DispatchTableError, match="duplicate arena"):
        parse_dispatch_table(payload)


def test_duplicate_row_is_refused():
    payload = _table()
    payload["rows"].append(copy.deepcopy(payload["rows"][0]))
    with pytest.raises(DispatchTableError, match="duplicate row"):
        parse_dispatch_table(payload)


def test_absolute_claiming_statistic_without_a_value_is_refused():
    payload = _table()
    payload["arenas"][0]["absolute_value"] = None
    with pytest.raises(DispatchTableError, match="absolute_value"):
        parse_dispatch_table(payload)


def test_ratio_only_arena_is_accepted_and_not_slo_eligible():
    payload = _table()
    payload["arenas"][0]["absolute_value"] = None
    payload["arenas"][0]["statistic"] = "ratio_only_no_absolute"
    payload["arenas"][0]["metric"] = "operator_ms"
    table = parse_dispatch_table(payload)
    arena = table.arena("prefill", "dense")
    assert arena.absolute_value is None
    assert not arena.slo_eligible
    assert arena.reference_ms() is None


def test_operator_ms_arena_with_an_absolute_is_still_not_slo_eligible():
    """Policy §5: raw standalone kernel timing is never served evidence."""
    payload = _table()
    payload["arenas"][0]["metric"] = "operator_ms"
    payload["arenas"][0]["statistic"] = "median_of_repeated_samples"
    table = parse_dispatch_table(payload)
    assert not table.arena("prefill", "dense").slo_eligible


def test_load_is_deterministically_ordered(tmp_path):
    payload = _table()
    payload["rows"].append({
        "format_family": "AAA", "phase": "prefill", "m_regime": "dense",
        "lane": "native", "relative_unit_cost": 2.0,
        "provenance": dict(_PROV),
    })
    p = tmp_path / "t.json"
    p.write_text(json.dumps(payload))
    first = load_dispatch_table(p)
    second = load_dispatch_table(p)
    assert [r.key for r in first.rows] == [r.key for r in second.rows]
    assert [r.key for r in first.rows] == sorted(r.key for r in first.rows)


def test_decode_tok_s_reference_converts_to_ms_per_token():
    """The one documented conversion, exact and named (assumption A7)."""
    payload = _table()
    payload["arenas"][0].update({
        "phase": "decode", "metric": "decode_tok_s", "absolute_value": 10.0,
    })
    payload["rows"][0]["phase"] = "decode"
    table = parse_dispatch_table(payload)
    assert table.arena("decode", "dense").reference_ms() == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 3. Format -> dispatch family
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt,family", [
    ("FP8_CB_K36", "FP8_CB"),
    ("FP8_CB_K28", "FP8_CB"),
    ("NVFP4_CB_K16", "NVFP4_CB"),
    ("NVFP4", "NVFP4"),
    ("FP8_E4M3", "FP8_E4M3"),
    ("BF16", "BF16"),
])
def test_dispatch_family_keys(fmt, family):
    assert dispatch_family_for_format(fmt) == family


def test_bf16_and_fp8_do_not_share_a_dispatch_family():
    """``format_registry``'s coarse family puts both in 'fp'; at serve time
    they have nothing in common, so the dispatch key must separate them."""
    from prismaquant import format_registry as fr
    assert fr.get_format("BF16").family == fr.get_format("FP8_E4M3").family
    assert (dispatch_family_for_format("BF16")
            != dispatch_family_for_format("FP8_E4M3"))


# ---------------------------------------------------------------------------
# 4. The shipped example table
# ---------------------------------------------------------------------------
def test_example_table_loads_and_every_row_cites_a_source():
    table = load_dispatch_table(example_table_path())
    assert table.rows
    for row in table.rows:
        assert row.provenance.source.strip()
        assert "gridbook" in row.provenance.source.lower()
        assert row.provenance.derivation.strip()
    for arena in table.arenas:
        assert arena.provenance.source.strip()


def test_example_table_declares_itself_proposal_data():
    table = load_dispatch_table(example_table_path())
    assert "proposal" in table.status
    assert "example" in table.status
    assert "PROPOSAL DATA" in table.description


def test_example_table_carries_the_audits_headline_prefill_tax():
    """The 1.44× dense-prefill row is the trade P5c exists to surface."""
    table = load_dispatch_table(example_table_path())
    row = table.row("FP8_CB", "prefill", "dense_prefill_1400", "fallback")
    assert row is not None
    assert row.relative_unit_cost == pytest.approx(1.44)
    assert "1.075" in row.provenance.measured_quantity
    assert "0.746" in row.provenance.measured_quantity


def test_example_table_prices_decode_at_parity():
    table = load_dispatch_table(example_table_path())
    row = table.row("FP8_CB", "decode", "decode_batch1", "fallback")
    assert row.relative_unit_cost == pytest.approx(0.999)
    # The slowest end of the published band, so the row cannot flatter CB.
    assert "10.27" in row.provenance.measured_quantity


def test_example_table_has_no_whole_model_nvfp4_cb_row():
    """The honest hole: no published whole-artifact number exists for the
    default fp4-CB BF16-bridge quality path, so the table has no row and the
    evaluator must refuse rather than interpolate."""
    table = load_dispatch_table(example_table_path())
    assert table.row("NVFP4_CB", "prefill", "dense_prefill_1400",
                     "fallback") is None
    assert table.row("NVFP4_CB", "decode", "decode_batch1", "fallback") is None
    assert missing_family_report(
        ["NVFP4_CB"], table, phase="prefill", m_regime="dense_prefill_1400",
    ) == ["NVFP4_CB"]
    # ...and it says so, loudly, in the table's own notes.
    assert any("NVFP4_CB HAS NO WHOLE-MODEL ROW" in n for n in table.notes)


def test_example_table_fused_mid_m_arenas_can_never_satisfy_an_slo():
    table = load_dispatch_table(example_table_path())
    for regime in ("dense_mid_m_32", "dense_mid_m_64", "dense_mid_m_128"):
        arena = table.arena("prefill", regime)
        assert arena is not None
        assert not arena.slo_eligible
        assert arena.metric == "operator_ms"


def test_example_table_fused_lane_rows_match_the_published_speedups():
    table = load_dispatch_table(example_table_path())
    for regime, speedup in (("dense_mid_m_32", 1.04),
                            ("dense_mid_m_64", 1.26),
                            ("dense_mid_m_128", 1.45)):
        row = table.row("FP8_CB", "prefill", regime, "fused_mid_m")
        assert row.relative_unit_cost == pytest.approx(1.0 / speedup, abs=1e-4)
        assert table.row("FP8_CB", "prefill", regime,
                         "fallback").relative_unit_cost == 1.0
