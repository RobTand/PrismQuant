"""R16 — `LaneSpec` + the GGUF KL evaluator adapter.

No server, no GPU, no binary: the lane declarations are data and the adapter's
testable half is a pure parse over CANNED `llama-perplexity` output. The live
path (actually invoking the binary) is integration and is not exercised here.
"""
import json
import math

import pytest

from prismaquant.gguf_kl_evaluator import (
    LlamaPerplexityParseError,
    build_llama_perplexity_command,
    frontier_row,
    kl_stats_from_parsed,
    measure_assignment_kl,
    parse_llama_perplexity_kl,
)
from prismaquant.lane_spec import (
    all_lane_specs,
    lane_gate_report,
    lane_spec_for_container,
    lane_spec_names,
    load_lane_spec,
)

# --------------------------------------------------------------------------
# Canned harness output — the CURRENT llama.cpp spelling ("Mean    KLD:") ...
# --------------------------------------------------------------------------
CANNED_CURRENT = """
perplexity: calculating perplexity over 64 chunks

====== Perplexity statistics ======
Mean PPL(Q)                   :   6.283551 ±   0.037587
Mean PPL(base)                :   6.229723 ±   0.037241
Cor(ln(PPL(Q)), ln(PPL(base))):  99.79%
Mean ln(PPL(Q)/PPL(base))     :   0.008607 ±   0.000284
Mean PPL(Q)/PPL(base)         :   1.008644 ±   0.000287
Mean PPL(Q)-PPL(base)         :   0.053828 ±   0.001800

====== KL divergence statistics ======
Mean    KLD:   0.011493 ±   0.000079
Maximum KLD:   3.945760
99.9%   KLD:   0.315907
99.0%   KLD:   0.103395
Median  KLD:   0.005226
10.0%  KLD:   0.000315
 5.0%  KLD:   0.000133
 1.0%  KLD:   0.000018
Minimum KLD:  -0.000000

====== Token probability statistics ======
Mean    Δp:  0.031 ±  0.005 %
Maximum Δp: 87.006%
RMS Δp    :  2.545 ± 0.010 %
Same top p: 98.316 ± 0.011 %
"""

# ... and the OLDER spelling, which archived logs still carry.
CANNED_LEGACY = """
====== Perplexity statistics ======
Mean PPL(Q)                   :   9.123456 ±   0.100000
Mean PPL(base)                :   8.987654 ±   0.090000

====== KL divergence statistics ======
Mean    KL divergence         :   0.021234 +/-   0.000456
Maximum KL divergence         :   1.234567
99.9%   KL divergence         :   0.456789
99.0%   KL divergence         :   0.123456
Median  KL divergence         :   0.005678
Minimum KL divergence         :   0.000012

====== Token probability statistics ======
Same top p: 85.123 +/- 0.234 %
"""


def test_parses_the_current_spelling():
    parsed = parse_llama_perplexity_kl(CANNED_CURRENT)
    assert parsed["kl_mean"] == pytest.approx(0.011493)
    assert parsed["kl_stderr"] == pytest.approx(0.000079)
    assert parsed["kl_max"] == pytest.approx(3.945760)
    assert parsed["kl_p99"] == pytest.approx(0.103395)
    assert parsed["kl_p999"] == pytest.approx(0.315907)
    assert parsed["kl_median"] == pytest.approx(0.005226)
    assert parsed["kl_min"] == pytest.approx(-0.0)
    assert parsed["ppl_q"] == pytest.approx(6.283551)
    assert parsed["ppl_base"] == pytest.approx(6.229723)
    assert parsed["top1_agreement_pct"] == pytest.approx(98.316)
    assert parsed["top1_agreement_stderr_pct"] == pytest.approx(0.011)


def test_parses_the_legacy_spelling_and_plusminus_variant():
    parsed = parse_llama_perplexity_kl(CANNED_LEGACY)
    assert parsed["kl_mean"] == pytest.approx(0.021234)
    assert parsed["kl_stderr"] == pytest.approx(0.000456)
    assert parsed["kl_max"] == pytest.approx(1.234567)
    assert parsed["top1_agreement_pct"] == pytest.approx(85.123)


def test_missing_kl_block_is_an_error_not_a_zero():
    with pytest.raises(LlamaPerplexityParseError):
        parse_llama_perplexity_kl("perplexity: 12 chunks\nFinal estimate: PPL = 6.3\n")


def test_stats_use_the_gold_lane_key_names():
    stats = kl_stats_from_parsed(parse_llama_perplexity_kl(CANNED_CURRENT))
    for key in ("kl_mean", "kl_stderr", "kl_p99", "kl_max", "nll_mean"):
        assert key in stats, key
    # NLL is ln(PPL) by definition — a rename, not a second measurement.
    assert stats["nll_mean"] == pytest.approx(math.log(6.283551), rel=1e-9)
    # The honest label: these quantiles are over TOKENS and are the harness's.
    assert stats["kl_tail_domain"] == "aggregate"
    assert stats["kl_evaluator"] == "llama_perplexity"


def test_measure_assignment_kl_matches_the_resident_interface():
    mean, per_seq, stats = measure_assignment_kl(
        model="artifact.gguf", base_logits="base.bin",
        output_text=CANNED_CURRENT,
    )
    assert mean == pytest.approx(0.011493)
    assert per_seq == []          # aggregate harness: no per-sequence values
    assert stats["kl_mean"] == mean


def test_frontier_row_emits_only_selector_columns():
    _, _, stats = measure_assignment_kl(
        model="a.gguf", base_logits="b.bin", output_text=CANNED_CURRENT)
    row = frontier_row("q4k-2.95", 2.95, stats)
    assert row["label"] == "q4k-2.95"
    assert row["bpp"] == pytest.approx(2.95)
    assert row["kl"] == pytest.approx(0.011493)
    assert row["kl_p99"] == pytest.approx(0.103395)


def test_command_matches_the_documented_lane_harness():
    cmd = build_llama_perplexity_command(
        model="/m/exported.gguf", base_logits="/m/base_logits.bin",
        corpus="/m/wiki.test.raw", chunks=64,
    )
    assert "--kl-divergence-base" in cmd and "--kl-divergence" in cmd
    assert cmd[cmd.index("--kl-divergence-base") + 1] == "/m/base_logits.bin"
    assert cmd[cmd.index("-f") + 1] == "/m/wiki.test.raw"
    assert cmd[cmd.index("--chunks") + 1] == "64"


# --------------------------------------------------------------------------
# LaneSpec
# --------------------------------------------------------------------------
def test_the_four_lanes_are_declared():
    assert set(lane_spec_names()) == {"compressed_tensors", "gguf", "gridbook_trellis", "nvfp4_cb"}
    containers = {s.export_container for s in all_lane_specs()}
    assert containers == {"compressed-tensors", "gguf", "gridbook_trellis", "nvfp4_cb"}


@pytest.mark.parametrize("container", ["compressed-tensors", "gguf", "gridbook_trellis", "nvfp4_cb"])
def test_every_lane_declares_the_four_things(container):
    spec = lane_spec_for_container(container)
    assert spec.endpoint.kind in {"openai", "llama_server", "none"}
    assert spec.kl_evaluator.entrypoint
    assert spec.gates, "a lane with no declared gates has no bar"
    assert spec.serve_scripts or spec.serve_command


def test_gates_are_advisory_and_that_is_deliberate():
    """The open half of R16's verdict — advisory vs blocking — is deferred to
    Robert. Pinned so a flip is an edit, not a drift."""
    for spec in all_lane_specs():
        assert spec.advisory_gates is True


def test_declared_serve_scripts_exist():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for spec in all_lane_specs():
        for script in spec.serve_scripts:
            assert (root / script).is_file(), f"{spec.id}: missing {script}"


def test_shipcard_slots_are_real_slots():
    from prismaquant.shipcard import ALL_SLOTS

    for spec in all_lane_specs():
        for slot in spec.shipcard_slots():
            assert slot in ALL_SLOTS, f"{spec.id}: unknown slot {slot}"


def test_gate_report_reads_shipcard_fill():
    spec = lane_spec_for_container("gguf")
    rows = lane_gate_report(spec, {"slots": {"gold.kl": {"kl": 0.01}}})
    by_slot = {r["shipcard_slot"]: r for r in rows}
    assert by_slot["gold.kl"]["filled"] is True
    assert by_slot["ship_gate"]["filled"] is False
    assert all(r["advisory"] for r in rows)


def test_serve_command_placeholder_substitution_is_total():
    spec = lane_spec_for_container("gguf")
    cmd = spec.render_serve_command(LLAMA_CPP_DIR="/llama", MODEL="/m/a.gguf")
    assert cmd[0] == "/llama/build/bin/llama-server"
    assert "/m/a.gguf" in cmd
    with pytest.raises(KeyError):
        spec.render_serve_command(MODEL="/m/a.gguf")


def test_lane_specs_are_valid_json_with_the_pinned_schema():
    from pathlib import Path

    spec_dir = Path(__file__).resolve().parents[1] / "prismaquant" / "lane_specs"
    for path in spec_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == "prismaquant.lane_spec.v1"
        assert payload["id"] == path.stem


def test_gguf_lane_names_the_llama_perplexity_adapter():
    spec = lane_spec_for_container("gguf")
    assert spec.kl_evaluator.kind == "llama_perplexity"
    module, _, attr = spec.kl_evaluator.entrypoint.partition(":")
    mod = __import__(module, fromlist=[attr])
    assert callable(getattr(mod, attr))


def test_native_and_cb_share_the_endpoint_agnostic_ship_gate():
    """The R16 finding in one assertion: the CB half is wiring, not
    capability — both lanes run the SAME ship-gate script on the SAME
    endpoint kind."""
    native = load_lane_spec("compressed_tensors")
    cb = load_lane_spec("nvfp4_cb")
    assert native.endpoint.kind == cb.endpoint.kind == "openai"
    assert (native.gate("ship_gate.ppl_p99nll").runner
            == cb.gate("ship_gate.ppl_p99nll").runner)


def test_cb_gold_ppl_runner_names_offline_wikitext_input():
    cb = load_lane_spec("nvfp4_cb")
    runner = cb.gate("gold.ppl").runner
    assert "--dsv4-gridbook-contract" in runner
    assert "--wikitext-inputs <gold-inputs.json>" in runner
