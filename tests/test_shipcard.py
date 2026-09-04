"""The ship record's refusal contract (R13).

The point of `shipcard.json` is that it says NO by default: an artifact whose
serve-lane slots were never closed must not read as shippable. These tests pin
the four ways it says no — unfilled, wrong build, failed check, spec-decode
tainted gold number — plus the fill path the serve-lane tools use.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from functools import lru_cache

import pytest

if not (pathlib.Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

from prismaquant.shipcard import (

    GOLD_SLOTS,
    REQUIRED_SLOTS,
    SHIPCARD_RESERVED_BYTES,
    artifact_bytes,
    build_shipcard,
    compute_model_sha,
    fill_slot,
    kv_shared_fisher_echo,
    load_shipcard,
    make_record,
    required_slots,
    unfilled_slots,
    verify,
    write_shipcard,
)
from prismaquant.shipcard_cli import main as shipcard_cli
from prismaquant.validate_quantized_model import (
    BOUNDARY_ENDPOINT,
    BOUNDARY_REQUEST_SCHEMA,
    BOUNDARY_RESPONSE_SCHEMA,
    DEFAULT_BOUNDARY_MAX_TOKENS,
    DEFAULT_BOUNDARY_REPS,
    DEFAULT_BOUNDARY_TEMPERATURE,
    DEFAULT_MAX_BOUNDARY_DEFECTS,
    DEFAULT_MAX_MEAN_NLL,
    DEFAULT_MAX_P99_NLL,
    DEFAULT_MAX_PPL,
    DEFAULT_MIN_GEN_LEN,
    DEFAULT_MIN_MTP_ACCEPT_P0,
)


def _artifact(
    tmp_path, *, name="exported", weight_bytes=b"weights", model_type="qwen3",
):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": model_type}))
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(weight_bytes)
    return model_dir


def _open_card(tmp_path, model_dir):
    card = build_shipcard(model_dir, build={"achieved_bpp": {"value": 4.75}})
    path = model_dir / "shipcard.json"
    write_shipcard(path, card)
    return path


_FAKE_FINGERPRINT = "f" * 64
_FAKE_COMMIT = "a" * 40

#: What a real gold record carries. These helpers used to fill the gold slots
#: with an EMPTY metrics dict and the card verified anyway, because
#: `_verify_gold_record` only ran on the Gridbook CB lane — so the tests were
#: encoding the hole rather than catching it. Every generic gold requirement
#: (finite metric, serve fingerprint, producer commit, position count,
#: score_positions=all) now applies on every lane, and the fixtures have to
#: look like real measurements.
_GOLD_METRICS = {
    "gold.kl": {
        "kl_mean": 0.0151,
        "kl_confident_mean": 0.0143,
        "n_positions": 4088,
        "n_samples": 8,
        "seqlen": 512,
        "score_positions": "all",
    },
    "gold.ppl": {
        "ppl": 8.33,
        "mean_nll": 2.12,
        "n_tokens_scored": 8192,
    },
}


def _fill_all(path, model_sha, *, spec=False, passed=True):
    for slot in REQUIRED_SLOTS:
        if slot == "ship_gate":
            fill_slot(path, slot, _ship_gate_record(
                model_sha, passed=passed, source=str(path.parent)))
            continue
        if slot.startswith("native_export."):
            # A passing smoke stamps what it ran and what it generated; the
            # exception path stamps no generation evidence. Both shapes must
            # replay through `_verify_native_export_record`.
            arm = slot.split(".")[1]
            metrics = {"arm": arm, "enforce_eager": arm == "eager"}
            if passed:
                metrics.update(
                    {"generated_chars": 128, "max_new_tokens": 16})
            fill_slot(path, slot, make_record(
                slot=slot, tool="validate_native_export.py", passed=passed,
                model_sha=model_sha, metrics=metrics,
                detail=f"{arm} smoke", git_commit=_FAKE_COMMIT))
            continue
        is_gold = slot in GOLD_SLOTS
        fill_slot(path, slot, make_record(
            slot=slot, tool="test", passed=passed, model_sha=model_sha,
            spec_decode_detected=(spec if is_gold else None),
            metrics=(_GOLD_METRICS.get(slot) if is_gold else None),
            serve_fingerprint=(_FAKE_FINGERPRINT if is_gold else None),
            git_commit=(_FAKE_COMMIT if is_gold else None),
        ))


_REFUSED_URL = "http://127.0.0.1:1"


@lru_cache(maxsize=1)
def _producer_check_names() -> frozenset:
    """The ship-gate ledger's key set, derived from the producer that owns it.

    Each check constructor names its own CheckResult without needing a live
    server: against a refused localhost port every probe fails fast and
    returns its (failing) verdict, whose name is what `run_validation` files
    under. A sixth check added to the producer appears here on its own; a
    roster restated from today's five names would not.
    """
    from prismaquant import validate_quantized_model as vqm

    return frozenset({
        vqm.check_serve_ready(_REFUSED_URL).name,
        vqm.check_generation_sanity(
            _REFUSED_URL, "probe", DEFAULT_MIN_GEN_LEN).name,
        vqm.check_perplexity(
            _REFUSED_URL, "probe",
            DEFAULT_MAX_PPL, DEFAULT_MAX_P99_NLL, DEFAULT_MAX_MEAN_NLL).name,
        vqm.check_mtp_acceptance(_REFUSED_URL, DEFAULT_MIN_MTP_ACCEPT_P0).name,
        vqm.check_boundary_behavior(_REFUSED_URL, "probe").name,
    })


def _ship_gate_record(model_sha, *, passed=True, source="artifact",
                      spec_decode_detected=False, detail=None):
    """A ship_gate record shaped like a real validator verdict.

    Thresholds are derived from the producer's own DEFAULT_* constants, not
    restated: if the catastrophic bounds move, this fixture moves with them
    and only the verifier's independent replay may refuse. Likewise the
    ledger's key set comes from `_producer_check_names`, so a check the
    producer adds is required here without anyone retyping a roster.
    """
    ledger = {name: {"passed": True} for name in _producer_check_names()}
    ledger["perplexity"] = {
        "passed": True,
        # Well clear of the catastrophic bounds (not near any limit).
        "perplexity": 8.33,
        "mean_nll_per_tok": 2.12,
        "max_nll_per_tok": 4.50,
        "n_tokens": 8192,
        "spec_decode_detected": False,
    }
    if "boundary_behavior" in ledger:
        # A bare {"passed": True} is the old argmax-blind shape wearing the
        # new name: the replay requires sampled generations at temp > 0 with
        # zero defects, so the fixture must look like a real measurement.
        ledger["boundary_behavior"] = {
            "passed": True,
            "endpoint": BOUNDARY_ENDPOINT,
            "request_schema": BOUNDARY_REQUEST_SCHEMA,
            "response_schema": BOUNDARY_RESPONSE_SCHEMA,
            "n_prompts": 5,
            "reps": 6,
            "n_generations": 30,
            "n_defects": 0,
            "max_defects": 0,
            "temperature": 1.0,
            "max_tokens": 64,
            "defects_by_kind": {
                "zero_tag": 0, "think_stutter": 0, "cap_truncation": 0},
            "failing_examples": [],
        }
    if detail is None:
        detail = ("serve_ready=pass; generation_sanity=pass; "
                  "perplexity=pass; mtp_acceptance=pass; "
                  "boundary_behavior=pass")
    return make_record(
        slot="ship_gate", tool="validate_quantized_model.py", passed=passed,
        model_sha=model_sha, metrics=ledger,
        detail=detail,
        spec_decode_detected=spec_decode_detected,
        git_commit=_FAKE_COMMIT,
        extra={
            "base_url": "http://127.0.0.1:8000",
            "served_model_name": "probe-artifact",
            "thresholds": {
                "max_ppl": DEFAULT_MAX_PPL,
                "max_mean_nll": DEFAULT_MAX_MEAN_NLL,
                "max_p99_nll": DEFAULT_MAX_P99_NLL,
                "min_gen_len": DEFAULT_MIN_GEN_LEN,
                "min_mtp_accept_p0": DEFAULT_MIN_MTP_ACCEPT_P0,
                "max_boundary_defects": DEFAULT_MAX_BOUNDARY_DEFECTS,
                "boundary_temperature": DEFAULT_BOUNDARY_TEMPERATURE,
                "boundary_max_tokens": DEFAULT_BOUNDARY_MAX_TOKENS,
                "boundary_reps": DEFAULT_BOUNDARY_REPS,
                "bos_token": None,
                "add_special_tokens": True,
            },
            "model_sha_source": source,
        },
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def test_model_sha_is_stable_and_content_sensitive(tmp_path):
    a = _artifact(tmp_path, name="a")
    b = _artifact(tmp_path, name="b")
    assert compute_model_sha(a) == compute_model_sha(a)
    assert compute_model_sha(a) == compute_model_sha(b), (
        "identical bytes and layout must hash identically — a copied artifact "
        "keeps its identity")

    (b / "config.json").write_text('{"model_type": "qwen3", "x": 1}')
    assert compute_model_sha(a) != compute_model_sha(b)

    c = _artifact(tmp_path, name="c", weight_bytes=b"weights-but-longer")
    assert compute_model_sha(a) != compute_model_sha(c)


def test_native_card_remains_verifiable_after_legitimate_copy(tmp_path):
    import shutil

    source = _artifact(tmp_path, name="native-source")
    source_card = _open_card(tmp_path, source)
    assert "weight_stat_attestation" not in load_shipcard(source_card)

    copied = tmp_path / "native-copy"
    shutil.copytree(source, copied)
    problems = verify(load_shipcard(copied / "shipcard.json"), model_dir=copied)
    assert all("artifact changed" not in problem for problem in problems)


def test_cb_identity_binds_canonical_config_and_codebook_not_inventory(
    tmp_path,
):
    model_dir = _artifact(tmp_path)
    codebook = model_dir / "cb_codebooks.pqcb"
    codebook.write_bytes(b"codebook-A")
    quant_config = {
        "config_groups": {
            "cb": {
                "scheme": {"grid": "fp4", "k": 16},
                "targets": ["model.layers.0.mlp.up_proj"],
            }
        },
        "provenance": {
            "producer": "resident",
            "artifact_inventory": {"export_directory_bytes": 123},
        },
    }
    quant_path = model_dir / "quant_config.json"
    quant_path.write_text(json.dumps(quant_config, indent=2))
    baseline = compute_model_sha(model_dir)

    # Formatting and the self-sized final inventory are not model semantics.
    quant_config["provenance"]["artifact_inventory"] = {
        "export_directory_bytes": 987654,
        "file_bytes": {"quant_config.json": 456},
    }
    quant_path.write_text(json.dumps(
        quant_config, sort_keys=True, separators=(",", ":")
    ))
    assert compute_model_sha(model_dir) == baseline

    # Every other quant-config field remains identity-bearing.
    quant_config["config_groups"]["cb"]["scheme"]["k"] = 17
    quant_path.write_text(json.dumps(quant_config))
    assert compute_model_sha(model_dir) != baseline

    # Restore the config, then change same-length codebook bytes. Content, not
    # just the sidecar's size, must distinguish the served model.
    quant_config["config_groups"]["cb"]["scheme"]["k"] = 16
    quant_path.write_text(json.dumps(quant_config))
    codebook.write_bytes(b"codebook-B")
    assert compute_model_sha(model_dir) != baseline
    assert artifact_bytes(model_dir) == (
        (model_dir / "model-00001-of-00001.safetensors").stat().st_size
        + codebook.stat().st_size
    )


def test_cb_identity_binds_same_size_weight_content_via_export_manifest(tmp_path):
    from prismaquant.shipcard import build_weight_content_manifest

    model_dir = _artifact(tmp_path, weight_bytes=b"weights-A")
    quant_config = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "provenance": {
            "weight_content_manifest": build_weight_content_manifest(model_dir),
        },
    }
    (model_dir / "quant_config.json").write_text(json.dumps(quant_config))
    baseline = compute_model_sha(model_dir)

    # The immutable content claim is part of model_sha even though routine
    # verification need not reread the large shard. A changed same-size shard
    # trips the card's cheap stat attestation immediately.
    path = _open_card(tmp_path, model_dir)
    weight = model_dir / "model-00001-of-00001.safetensors"
    weight.write_bytes(b"weights-B")
    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any("artifact changed since the shipcard was opened" in p for p in problems)
    assert compute_model_sha(model_dir) == baseline


def test_cb_identity_binds_auxiliary_serving_files(tmp_path):
    from prismaquant.shipcard import build_weight_content_manifest

    model_dir = _artifact(tmp_path)
    tokenizer = model_dir / "tokenizer.json"
    tokenizer.write_bytes(b"tokenizer-A")
    quant_config = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "provenance": {
            "weight_content_manifest": build_weight_content_manifest(model_dir),
        },
    }
    (model_dir / "quant_config.json").write_text(json.dumps(quant_config))
    baseline = compute_model_sha(model_dir)
    tokenizer.write_bytes(b"tokenizer-B")
    assert compute_model_sha(model_dir) != baseline


def test_reattest_accepts_copy_but_refuses_changed_weight_content(tmp_path):
    import shutil

    from prismaquant.shipcard import (
        build_weight_content_manifest,
        reattest_weight_stats,
    )

    source = _artifact(tmp_path, name="source", weight_bytes=b"weights-A")
    quant_config = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "provenance": {
            "weight_content_manifest": build_weight_content_manifest(source),
        },
    }
    (source / "quant_config.json").write_text(json.dumps(quant_config))
    source_card = _open_card(tmp_path, source)

    copied = tmp_path / "copied"
    shutil.copytree(source, copied)
    copied_card = copied / source_card.name
    assert verify(load_shipcard(copied_card), model_dir=copied)
    reattest_weight_stats(copied_card, copied)
    assert not any(
        "artifact changed since the shipcard was opened" in problem
        for problem in verify(load_shipcard(copied_card), model_dir=copied)
    )

    (copied / "model-00001-of-00001.safetensors").write_bytes(b"weights-B")
    with pytest.raises(ValueError, match="content differs"):
        reattest_weight_stats(copied_card, copied)


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------
def test_fresh_card_refuses_every_slot(tmp_path):
    model_dir = _artifact(tmp_path)
    card = build_shipcard(model_dir, build={})
    assert unfilled_slots(card) == list(REQUIRED_SLOTS)
    problems = verify(card, model_dir=model_dir)
    assert len(problems) == len(REQUIRED_SLOTS)
    assert all("UNFILLED" in p for p in problems)


def _gold_problems(model_dir, path, slot):
    _fill_all(path, compute_model_sha(model_dir))
    return [p for p in verify(load_shipcard(path), model_dir=model_dir)
            if p.startswith(f"{slot}:")]


def test_full_card_verifies(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))
    assert verify(load_shipcard(path), model_dir=model_dir) == []


def test_shipcard_fixed_reservation_survives_every_slot_fill(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    assert path.stat().st_size == SHIPCARD_RESERVED_BYTES

    model_sha = compute_model_sha(model_dir)
    for slot in REQUIRED_SLOTS:
        is_gold = slot in GOLD_SLOTS
        if slot == "ship_gate":
            # The ledger's key set is contractual: padding rides in `detail`,
            # never as an extra metrics key.
            record = _ship_gate_record(
                model_sha, source=str(model_dir), detail="x" * 4096)
        elif slot.startswith("native_export."):
            arm = slot.split(".")[1]
            record = make_record(
                slot=slot,
                tool="validate_native_export.py",
                passed=True,
                model_sha=model_sha,
                metrics={"arm": arm, "generated_chars": 128,
                         "enforce_eager": arm == "eager",
                         "max_new_tokens": 16},
                detail="x" * 4096,
                git_commit=_FAKE_COMMIT,
            )
        else:
            record = make_record(
                slot=slot,
                tool="fixed-size-test",
                passed=True,
                model_sha=model_sha,
                metrics={"detail": "x" * 4096, **_GOLD_METRICS.get(slot, {})},
                spec_decode_detected=False if is_gold else None,
                serve_fingerprint=(_FAKE_FINGERPRINT if is_gold else None),
                git_commit=(_FAKE_COMMIT if is_gold else None),
            )
        fill_slot(path, slot, record)
        assert path.stat().st_size == SHIPCARD_RESERVED_BYTES

    assert verify(load_shipcard(path), model_dir=model_dir) == []


def test_shipcard_reservation_overflow_preserves_previous_record(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    before = path.read_bytes()
    card = load_shipcard(path)
    card["build"]["oversized"] = "x" * SHIPCARD_RESERVED_BYTES

    with pytest.raises(ValueError, match="fixed reservation"):
        write_shipcard(path, card)

    assert path.read_bytes() == before


def test_shipcard_write_falls_back_to_compact_before_refusing(tmp_path):
    """indent=2 inflates the card ~1.5x; a card that only overflows the
    reservation pretty-printed must be written compact, not dropped.  The
    six-slot DSv4 card crossed this line on its LAST slot fill (the graph
    endpoint record, 2026-08-18) after every gate had already passed."""
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    card = load_shipcard(path)
    reserved = card["reserved_file_bytes"]
    # Grow the card until pretty-printed exceeds the reservation but compact
    # still fits: many small fields inflate most under indentation.
    filler = {f"k{i}": i for i in range(reserved // 18)}
    card["build"]["filler"] = filler
    pretty = len(json.dumps(card, indent=2, default=str)) + 1
    compact = len(json.dumps(card, separators=(",", ":"), default=str)) + 1
    assert pretty > reserved > compact, (pretty, reserved, compact)

    write_shipcard(path, card)

    raw = path.read_bytes()
    assert len(raw) == reserved, "the byte contract must hold exactly"
    assert load_shipcard(path)["build"]["filler"] == filler


def test_record_from_another_build_is_refused(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, "deadbeef" * 8)
    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert problems and all("another build" in p for p in problems)


def test_artifact_edited_after_the_card_was_opened_is_refused(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"re-exported!")

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any("artifact changed since the shipcard was opened" in p
               for p in problems)


def test_failed_record_is_refused(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))
    fill_slot(path, "ship_gate", _ship_gate_record(
        compute_model_sha(model_dir), passed=False,
        source=str(model_dir), detail="p99 NLL 9.4 > 6.0"))

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert problems == ["ship_gate: FAILED — p99 NLL 9.4 > 6.0"]


# ---------------------------------------------------------------------------
# ship_gate replay (#156): the receipt's threshold contract, check ledger,
# token evidence and endpoint binding are replayed by `verify`, not filed.
# Each test below fails while the replay is unwired (the mutated record
# verifies clean) and passes once it is.
# ---------------------------------------------------------------------------
def test_ship_gate_lowered_threshold_is_refused(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    sha = compute_model_sha(model_dir)
    _fill_all(path, sha)
    rec = _ship_gate_record(sha, source=str(model_dir))
    rec["thresholds"] = {
        **rec["thresholds"], "max_ppl": DEFAULT_MAX_PPL * 4}
    fill_slot(path, "ship_gate", rec)

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any("threshold max_ppl" in p for p in problems)


@pytest.mark.parametrize("field", [
    "base_url", "served_model_name", "model_sha_source"])
def test_ship_gate_missing_endpoint_binding_is_refused(tmp_path, field):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    sha = compute_model_sha(model_dir)
    _fill_all(path, sha)
    rec = _ship_gate_record(sha, source=str(model_dir))
    rec[field] = ""
    fill_slot(path, "ship_gate", rec)

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any(field in p for p in problems)


def test_ship_gate_incomplete_ledger_is_refused(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    sha = compute_model_sha(model_dir)
    _fill_all(path, sha)
    rec = _ship_gate_record(sha, source=str(model_dir))
    # Even the most-evidenced single check is not the ledger: keep only the
    # perplexity entry (named in the producer's own vocabulary) and drop the
    # rest of the derived key set.
    rec["metrics"] = {"perplexity": rec["metrics"]["perplexity"]}
    assert set(rec["metrics"]) != set(_producer_check_names())
    fill_slot(path, "ship_gate", rec)

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any("ledger is incomplete" in p for p in problems)


def test_ship_gate_unscored_perplexity_is_refused(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    sha = compute_model_sha(model_dir)
    _fill_all(path, sha)
    rec = _ship_gate_record(sha, source=str(model_dir))
    # The producer backfills a missing token count with 0
    # (validate_quantized_model.py); the replay must read that 0 as "no
    # evidence", not as a number.
    rec["metrics"]["perplexity"] = {
        **rec["metrics"]["perplexity"], "n_tokens": 0}
    fill_slot(path, "ship_gate", rec)

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any("scored no tokens" in p for p in problems)


@pytest.mark.parametrize("spec, expected", [
    (True, "is TRUE"),
    (None, "is unknown"),
])
def test_gold_slots_refuse_spec_decode_states(tmp_path, spec, expected):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))
    # Re-fill gold.kl as a fully valid record apart from the spec-decode
    # state, so the assertion below isolates the spec-decode refusal instead of
    # counting the generic gold-evidence problems alongside it.
    fill_slot(path, "gold.kl", make_record(
        slot="gold.kl", tool="test", passed=True,
        model_sha=compute_model_sha(model_dir), spec_decode_detected=spec,
        metrics=_GOLD_METRICS["gold.kl"],
        serve_fingerprint=_FAKE_FINGERPRINT, git_commit=_FAKE_COMMIT))

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert len(problems) == 1
    assert problems[0].startswith("gold.kl: spec_decode_detected")
    assert expected in problems[0]


def test_unknown_slot_is_rejected(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    with pytest.raises(KeyError):
        make_record(slot="gold.mmlu", tool="test", passed=True, model_sha="x")
    with pytest.raises(KeyError):
        fill_slot(path, "gold.mmlu", {"passed": True})


# ---------------------------------------------------------------------------
# Build-lane facts
# ---------------------------------------------------------------------------
def test_kv_shared_fisher_echo_flags_an_unvalidated_allocation():
    clean = kv_shared_fisher_echo({})
    assert clean["unvalidated_kv_fisher_correction"] is False

    overridden = kv_shared_fisher_echo(
        {"PRISMAQUANT_ALLOW_KV_SHARED_FISHER": "1"})
    assert overridden["unvalidated_kv_fisher_correction"] is True

    severed = kv_shared_fisher_echo({"PRISMAQUANT_KV_COTANGENT": "0"})
    assert severed["kv_cotangent_path_enabled"] is False
    assert severed["unvalidated_kv_fisher_correction"] is True


def test_export_writes_a_card_with_build_facts_and_empty_slots(tmp_path):
    """The exporter's `_write_shipcard`, without importing torch's world."""
    from prismaquant.export_native_compressed import _write_shipcard

    model_dir = _artifact(tmp_path)
    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({"model.layers.0.mlp.up_proj": {"bits": 4}}))
    (tmp_path / "pareto.knees.json").write_text(json.dumps({
        "primary": "log_error",
        "log_error": {"achieved_bits": 4.7513, "target_bits": 4.75},
    }))

    _write_shipcard(
        model_dir,
        source_model="/models/Qwen3-4B",
        layer_config_path=str(recipe),
        assignment={"model.layers.0.mlp.up_proj": "NVFP4"},
        config_assignment={"model.layers.0.mlp.up_proj": "NVFP4"},
        hist={("NVFP4", "packed"): 1},
    )

    card = load_shipcard(model_dir / "shipcard.json")
    assert unfilled_slots(card) == list(REQUIRED_SLOTS)
    build = card["build"]
    assert build["achieved_bpp"]["value"] == pytest.approx(4.7513)
    assert build["achieved_bpp"]["source"] == "pareto.knees.json:log_error"
    assert build["layer_config_sha"] and build["assignment_hash"]

    assert build["format_histogram"] == {"NVFP4/packed": 1}
    assert "PRISMAQUANT_GPTQ_DAMP" in build["render_levers"]
    assert "unvalidated_kv_fisher_correction" in build["kv_shared_fisher"]
    assert card["artifact_bytes"] == len(b"weights")


# ---------------------------------------------------------------------------
# CLI
def test_validated_selection_beats_the_surrogate_knee_file(tmp_path):
    """A validated recipe must not be labelled with the surrogate knee's bpp.

    Regression: on Qwen3.8-27B arm B the card claimed 5.9994 bpp
    (pareto.knees.json:log_error) for bytes that were the validated frontier's
    4.7496 pick -- a 1.25 bpp false claim on the publication gate.  Both the
    knee file and the recipe's own stale `achieved_bits` are present here, as
    they are on a real validated run, so this pins the precedence rather than
    just the happy path.
    """
    from prismaquant.shipcard import allocator_achieved_bpp

    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({
        "model.layers.0.mlp.up_proj": {"bits": 4},
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "achieved_bits": 5.50016116330051,      # stale: pre-selection
            "target_bits": 5.5,
            "selected_by": "validated_frontier:kneedle",
            "selected_achieved_bits": 4.749587350041043,
            "selected_label": "allocator_target_4p7500_achieved_4p7496",
        },
    }))
    (tmp_path / "pareto.knees.json").write_text(json.dumps({
        "primary": "log_error",
        "log_error": {"achieved_bits": 5.999404111844319, "target_bits": 6.0},
    }))

    got = allocator_achieved_bpp(recipe)
    assert got["value"] == pytest.approx(4.749587350041043)
    assert got["source"] == "layer_config.json:validated_frontier:kneedle"
    assert got["selected_label"] == "allocator_target_4p7500_achieved_4p7496"


def test_allocator_written_recipe_prefers_its_own_metadata(tmp_path):
    """No validated selection: the recipe's own achieved_bits still wins.

    It is coupled to this file; the knee file is a separate artifact that can
    describe a different point.
    """
    from prismaquant.shipcard import allocator_achieved_bpp

    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({
        "model.layers.0.mlp.up_proj": {"bits": 4},
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "achieved_bits": 4.7513,
            "target_bits": 4.75,
            "achieved_bits_scope": "body_assignment_tensor_payload",
        },
    }))
    (tmp_path / "pareto.knees.json").write_text(json.dumps({
        "primary": "log_error",
        "log_error": {"achieved_bits": 5.999404111844319, "target_bits": 6.0},
    }))

    got = allocator_achieved_bpp(recipe)
    assert got["value"] == pytest.approx(4.7513)
    assert got["source"] == "layer_config.json:achieved_bits"
    assert got["scope"] == "body_assignment_tensor_payload"


def test_validated_selection_without_a_bpp_reports_nothing(tmp_path):
    """Announcing a validated selection but stamping no bpp must not fall back.

    The knee file describes the surrogate frontier, i.e. some OTHER point, so
    silence is correct and a number would be a fabrication.
    """
    from prismaquant.shipcard import allocator_achieved_bpp

    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({
        "model.layers.0.mlp.up_proj": {"bits": 4},
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "selected_by": "validated_frontier:kneedle",
        },
    }))
    (tmp_path / "pareto.knees.json").write_text(json.dumps({
        "primary": "log_error",
        "log_error": {"achieved_bits": 5.999404111844319},
    }))

    got = allocator_achieved_bpp(recipe)
    assert got["value"] is None
    assert "no bpp stamped" in got["source"]


# ---------------------------------------------------------------------------
def test_cli_verify_exit_codes(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)

    assert shipcard_cli(["verify", str(path), "--model-dir", str(model_dir)]) == 1
    assert "REFUSED" in capsys.readouterr().out

    _fill_all(path, compute_model_sha(model_dir))
    assert shipcard_cli(["verify", str(path)]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_fill_from_a_gold_result_json(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    result = tmp_path / "kl_student.json"
    result.write_text(json.dumps({
        "model": str(model_dir),
        "kl_confident_mean": 0.0143,
        "kl_mean": 0.0201,
        "n_samples": 8,
        "n_positions": 4088,
        "seqlen": 512,
        # The gold KL contract is every prompt position. The fixture carried
        # no `score_positions` until 2026-09-02, when the fill-time replay
        # stopped being scoped to the retired Gridbook lane's cards and
        # started tracking `verify()` on every lane, as its own comment always
        # demanded. Without the field the CLI now (correctly) fills
        # passed=false, because publication would refuse the same record.
        "score_positions": "all",
        "spec_decode_detected": False,
        "serve_fingerprint": "f" * 64,
        "git_commit": "a" * 40,
    }))

    assert shipcard_cli([
        "fill", str(path), "--slot", "gold.kl", "--record", str(result)]) == 0

    record = load_shipcard(path)["slots"]["gold.kl"]
    assert record["passed"] is True
    assert record["model_sha"] == compute_model_sha(model_dir)
    assert record["metrics"]["kl_confident_mean"] == pytest.approx(0.0143)
    assert record["serve_fingerprint"] == "f" * 64
    assert record["git_commit"] == "a" * 40
    assert "gold.ppl" in capsys.readouterr().out  # still-unfilled list


def test_cli_fill_refuses_a_spec_decode_tainted_gold_record(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    result = tmp_path / "ppl.json"
    result.write_text(json.dumps({
        "model": str(model_dir), "ppl": 4.12, "spec_decode_detected": True}))

    assert shipcard_cli([
        "fill", str(path), "--slot", "gold.ppl", "--record", str(result)]) == 2
    assert "DRAFT model" in capsys.readouterr().err
    assert load_shipcard(path)["slots"]["gold.ppl"] is None

    # ...and an unknown detection is refused for the same reason.
    result.write_text(json.dumps({"model": str(model_dir), "ppl": 4.12}))
    assert shipcard_cli([
        "fill", str(path), "--slot", "gold.ppl", "--record", str(result)]) == 2

    # --allow-spec-decode records it, and verify still refuses.
    result.write_text(json.dumps({
        "model": str(model_dir), "ppl": 4.12, "spec_decode_detected": True}))
    assert shipcard_cli([
        "fill", str(path), "--slot", "gold.ppl", "--record", str(result),
        "--allow-spec-decode"]) == 0
    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any("gold.ppl: spec_decode_detected is TRUE" in p for p in problems)


def test_cli_show_lists_unfilled_slots(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    assert shipcard_cli(["show", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.count("UNFILLED") == len(REQUIRED_SLOTS)


@pytest.mark.parametrize("score_positions, expect_refusal", [
    ("all", False),
    ("final", True),
    (None, True),
])
def test_gold_kl_refuses_the_last_token_hook_screen(
    tmp_path, score_positions, expect_refusal
):
    """A positive sample count was never enough to make a number the gold KL.

    `measure_vllm_full_kl.py --score-positions final` — its DEFAULT — scores one
    position per sequence, the window-final context. That is the cheap
    last-token "hook KL" screen: triage only, never a promotion metric. It
    reports n_samples=8 and sailed through the count check, so the card could
    not tell an 8-position screen from a 4088-position gold measurement.

    Found 2026-08-14 on the Qwen3.8-27B lane, where the driver simply omitted
    the flag: the teacher wrote shape [8, 248077] — 8 positions, not 8x511.
    """
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))

    metrics = dict(_GOLD_METRICS["gold.kl"])
    if score_positions is None:
        metrics.pop("score_positions")
        # `final` mode does not stamp the key at all, which is why absence has
        # to be refused just as loudly as the explicit value.
        metrics["n_positions"] = 8
    else:
        metrics["score_positions"] = score_positions
    fill_slot(path, "gold.kl", make_record(
        slot="gold.kl", tool="test", passed=True,
        model_sha=compute_model_sha(model_dir), spec_decode_detected=False,
        metrics=metrics,
        serve_fingerprint=_FAKE_FINGERPRINT, git_commit=_FAKE_COMMIT))

    problems = verify(load_shipcard(path), model_dir=model_dir)
    hits = [p for p in problems if "score_positions" in p]
    assert bool(hits) is expect_refusal, problems
    if expect_refusal:
        assert "triage only" in hits[0]


def test_native_lane_gold_slots_are_verified_at_all(tmp_path):
    """The generic gold checks used to run only when the card was Gridbook CB.

    A NATIVE card — the default lane, and the one shipping artifacts — had its
    gold slots checked for nothing but spec-decode, so an empty metrics dict
    with no fingerprint and no producer commit verified clean.
    """
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    for slot in GOLD_SLOTS:
        fill_slot(path, slot, make_record(
            slot=slot, tool="test", passed=True,
            model_sha=compute_model_sha(model_dir),
            spec_decode_detected=False))

    problems = verify(load_shipcard(path), model_dir=model_dir)
    for slot in GOLD_SLOTS:
        assert any(
            p.startswith(f"{slot}: carries no finite") for p in problems
        ), problems
        assert any(
            p == f"{slot}: missing exact serve fingerprint" for p in problems
        ), problems
        assert any(
            p == f"{slot}: missing full producer git commit" for p in problems
        ), problems


def test_native_model_sha_attests_the_chat_template(tmp_path):
    """A native card used to bind config.json plus container SIZES only.

    Auxiliary files went unhashed unless the artifact carried a
    `quant_config.json` (i.e. unless it was a CB artifact), so swapping
    `chat_template.jinja` or `tokenizer.json` on a native checkpoint left
    `model_sha` bit-identical. Demonstrated 2026-08-15 on the published
    Qwen3.8-27B native artifact. That is not cosmetic on a tool-calling model:
    the chat template decides where a tool call is emitted, so a served
    artifact with the wrong one is broken in a way no weight check sees.
    """
    model_dir = _artifact(tmp_path)
    (model_dir / "chat_template.jinja").write_text("{{ messages }}")
    (model_dir / "tokenizer.json").write_text('{"version": "1"}')

    before = compute_model_sha(model_dir)
    (model_dir / "chat_template.jinja").write_text("{{ messages }}{# swap #}")
    assert compute_model_sha(model_dir) != before

    (model_dir / "chat_template.jinja").write_text("{{ messages }}")
    assert compute_model_sha(model_dir) == before
    (model_dir / "tokenizer.json").write_text('{"version": "2"}')
    assert compute_model_sha(model_dir) != before


def test_serving_an_artifact_does_not_invalidate_its_own_card(tmp_path):
    """`serve_manifest.json` is evidence ABOUT a serve, not artifact content.

    `scripts/lib/serve_manifest.sh` writes the R15 serve fingerprint INTO the
    model dir after a server comes up. It records the serving stack -- image,
    argv, the loaded `.so` set, hostname, boot id, timestamp -- so it differs
    between two serves of byte-identical weights. Hashing it made the act of
    VALIDATING an artifact invalidate the card that validation was for:
    observed 2026-08-15 on Qwen3.8-27B CB-A, where the eager smoke moved
    `model_sha` 677f278a -> bf6abc17 and `verify` then reported "artifact
    changed since the shipcard was opened".

    Nothing goes unbound: every slot that cites a manifest binds it by its own
    `*serve_manifest_sha256`, which is where a claim about a serve belongs.
    """
    model_dir = _artifact(tmp_path)
    (model_dir / "chat_template.jinja").write_text("{{ messages }}")

    path = _open_card(tmp_path, model_dir)
    before = compute_model_sha(model_dir)
    _fill_all(path, before)
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    (model_dir / "serve_manifest.json").write_text(
        '{"schema": "prismaquant.serve_manifest/1", "boot_id": "a"}')
    assert compute_model_sha(model_dir) == before
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    # A second serve writes a DIFFERENT manifest; the identity must not move.
    (model_dir / "serve_manifest.json").write_text(
        '{"schema": "prismaquant.serve_manifest/1", "boot_id": "b"}')
    assert compute_model_sha(model_dir) == before

    # The exclusion is by exact name, not by shape: a file that merely looks
    # like it stays attested.
    (model_dir / "serve_manifest.json.bak").write_text("{}")
    assert compute_model_sha(model_dir) != before


def test_documenting_an_artifact_does_not_invalidate_its_own_card(tmp_path):
    """`README.md` in a model dir IS the HF model card, not artifact content.

    `tools/publish_artifact.py` has no --model-card argument: it uploads the
    complete local file set with no filters, so the card reaches the Hub only
    by sitting in the artifact directory under that name. Hashing it made the
    act of DOCUMENTING an artifact invalidate the card that documents it --
    the same failure as the serve fingerprint above, one step earlier in the
    release. Observed 2026-08-15 on qwen38-27b-arm-b/exported: a README
    dropped in at 18:33 moved the identity off the 17:55 card
    (e7ac09f8 -> 3c4a83a1) and locked the artifact out of publication.

    It also decides whether an artifact can quote its own measured numbers.
    Every gate record binds `model_sha`; gold KL/PPL only exists after the
    gates; writing it into the card would invalidate the records that produced
    it. Re-running the gates does not escape that -- KL drifts across docker
    sessions -- so the exclusion is what makes a self-describing card possible.
    """
    model_dir = _artifact(tmp_path)
    (model_dir / "chat_template.jinja").write_text("{{ messages }}")

    path = _open_card(tmp_path, model_dir)
    before = compute_model_sha(model_dir)
    _fill_all(path, before)
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    (model_dir / "README.md").write_text("# a model card\n")
    assert compute_model_sha(model_dir) == before
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    # And the gold numbers can be written into it afterwards.
    (model_dir / "README.md").write_text("# a model card\n\nKL 0.0142\n")
    assert compute_model_sha(model_dir) == before
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    # Exact filenames, not a category: the card figures share the exclusion
    # (2026-08-18 -- they are rendered FROM the attested quant_config after
    # the gates by construction), and a doc under any other name stays
    # attested.
    (model_dir / "allocation-map.png").write_bytes(b"\x89PNG\r\n")
    assert compute_model_sha(model_dir) == before
    (model_dir / "byte-budget.png").write_bytes(b"\x89PNG\r\n")
    assert compute_model_sha(model_dir) == before
    assert verify(load_shipcard(path), model_dir=model_dir) == []
    (model_dir / "NOTES.md").write_text("notes")
    assert compute_model_sha(model_dir) != before


def test_a_card_stamped_while_the_figures_were_hashed_still_verifies(tmp_path):
    """Same tolerance as the README scope change, for the card figures.

    An artifact published with a hashed allocation-map.png keeps verifying via
    the `legacy_figures_hashed` fallback -- and keeps its figure attestation:
    tampering with the figure still moves the legacy identity.
    """
    model_dir = _artifact(tmp_path)
    (model_dir / "allocation-map.png").write_bytes(b"\x89PNG\r\nv1")

    legacy_sha = compute_model_sha(model_dir, legacy_figures_hashed=True)
    assert legacy_sha != compute_model_sha(model_dir)

    path = _open_card(tmp_path, model_dir)
    card = load_shipcard(path)
    card["model_sha"] = legacy_sha
    write_shipcard(path, card)
    _fill_all(path, legacy_sha)
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    (model_dir / "allocation-map.png").write_bytes(b"\x89PNG\r\ntampered")
    assert any("artifact changed" in p
               for p in verify(load_shipcard(path), model_dir=model_dir))


def test_a_card_stamped_while_the_readme_was_hashed_still_verifies(tmp_path):
    """The fix must not unbreak future artifacts by breaking present ones.

    A card written under the old scope on a directory that already contained a
    README verifies today only because the README was hashed into it. `verify`
    accepts that identity as a fallback, and only as a fallback.
    """
    model_dir = _artifact(tmp_path)
    (model_dir / "README.md").write_text("# an already-published card\n")

    legacy_sha = compute_model_sha(model_dir, legacy_readme_hashed=True)
    assert legacy_sha != compute_model_sha(model_dir)

    path = _open_card(tmp_path, model_dir)
    card = load_shipcard(path)
    card["model_sha"] = legacy_sha
    write_shipcard(path, card)
    _fill_all(path, legacy_sha)
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    # Still a fallback: editing that README moves the legacy identity too, so
    # a card from the hashed era keeps its README attestation.
    (model_dir / "README.md").write_text("# tampered\n")
    assert any("artifact changed" in p
               for p in verify(load_shipcard(path), model_dir=model_dir))


def test_a_card_written_under_the_legacy_native_scope_still_verifies(tmp_path):
    """Published native cards must not all read as 'artifact changed'.

    The legacy identity described its artifact faithfully under the rules it
    was computed with, so `verify` accepts it as a FALLBACK. It must remain a
    fallback: the legacy scope cannot produce a current-scope sha, so a card
    written today is never checked the weak way.
    """
    model_dir = _artifact(tmp_path)
    (model_dir / "chat_template.jinja").write_text("{{ messages }}")

    legacy_sha = compute_model_sha(model_dir, legacy_native_scope=True)
    assert legacy_sha != compute_model_sha(model_dir)

    path = _open_card(tmp_path, model_dir)
    card = load_shipcard(path)
    card["model_sha"] = legacy_sha
    write_shipcard(path, card)
    _fill_all(path, legacy_sha)

    assert verify(load_shipcard(path), model_dir=model_dir) == []

    # The legacy scope hashes container SIZES, so a same-size weight swap is
    # exactly what it could never see -- and still cannot. This test pins the
    # tolerance as bounded, not as a hole: what it accepts is the old
    # guarantee, never less.
    (model_dir / "chat_template.jinja").write_text("{{ messages }}{# swap #}")
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"longer!!")
    assert any(
        "artifact changed since the shipcard was opened" in problem
        for problem in verify(load_shipcard(path), model_dir=model_dir)
    )


def _cb_recipe(tmp_path, *, units, bytes_per_unit, params_per_unit, meta):
    """A CB recipe whose units carry the per-unit serialized price."""
    payload = {}
    for i in range(units):
        payload[f"model.layers.{i}.mlp.up_proj"] = {
            "bits": 4,
            "cb_serialized_identity": json.dumps({
                "format": "NVFP4_CB_K18",
                "params": params_per_unit,
                "tensor_payload_bytes": bytes_per_unit,
            }),
        }
    payload["__prismaquant__"] = meta
    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps(payload))
    return recipe


def test_recipe_priced_bpp_is_scope_matched_and_a_lower_bound(tmp_path):
    """Numerator and denominator come from the SAME entries, so no sidecar noise.

    K18 on (4096, 2048) is 2,654,212 payload bytes over 8,388,608 params --
    2,654,208 packed (2.53125 bpp exactly: 2.25 index + 0.28125 fp4 group
    scale) plus the 4-byte input global scale, so 2.5312538. These are the
    literal numbers the shipped DSv4 recipe's own identities declare.
    Units without a per-unit price are excluded from BOTH sums, which is what
    makes the result a lower bound rather than an estimate.
    """
    from prismaquant.shipcard import recipe_priced_bpp

    recipe = _cb_recipe(
        tmp_path, units=4, bytes_per_unit=2654212, params_per_unit=8388608,
        meta={"achieved_bits": 2.53125},
    )
    # One FP8_SOURCE passthrough Linear, priced by nobody.
    payload = json.loads(recipe.read_text())
    payload["model.layers.9.self_attn.wkv"] = {"bits": 8, "data_type": "fp8_e4m3"}
    recipe.write_text(json.dumps(payload))

    got = recipe_priced_bpp(recipe)
    assert got["value"] == pytest.approx(2.53125, rel=1e-5)
    assert got["priced_units"] == 4
    assert got["total_units"] == 5
    assert got["coverage_units"] == pytest.approx(0.8)


def test_recipe_priced_bpp_is_not_applicable_without_per_unit_prices(tmp_path):
    """A non-CB recipe declares no per-unit bytes; that is not a failure."""
    from prismaquant.shipcard import recipe_priced_bpp

    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({
        "model.layers.0.mlp.up_proj": {"bits": 4},
        "__prismaquant__": {"achieved_bits": 4.75},
    }))
    got = recipe_priced_bpp(recipe)
    assert got["value"] is None
    assert "no unit carries a per-unit serialized price" in got["reason"]


def test_achieved_bpp_cross_check_catches_a_label_describing_another_point(tmp_path):
    """The DSv4 `artifact-aura-cb-112p69` failure, reproduced.

    That card published 4.3065 bpp read from a sibling `pareto.knees.json`
    while the recipe it named priced to 2.7385 -- a 57% false public claim that
    would silently break every matched-bpp comparison built on it. No
    precedence rule catches it on its own, because the stale number is not the
    wrong FIELD, it is the right field describing the wrong POINT.
    """
    from prismaquant.shipcard import allocator_achieved_bpp, verify

    recipe = _cb_recipe(
        tmp_path, units=4, bytes_per_unit=2654212, params_per_unit=8388608,
        meta={"schema": "prismaquant.layer_config_meta.v1"},   # declares no bpp
    )
    (tmp_path / "pareto.knees.json").write_text(json.dumps({
        "primary": "log_error",
        "log_error": {"achieved_bits": 4.306515854190045, "target_bits": 4.5},
    }))

    got = allocator_achieved_bpp(recipe)
    assert got["value"] == pytest.approx(4.306515854190045)
    assert got["source"] == "pareto.knees.json:log_error"

    cross = got["cross_check"]
    assert cross["verdict"] == "DISAGREE"
    assert cross["recipe_priced_bpp"] == pytest.approx(2.53125, rel=1e-5)
    assert cross["relative_difference"] > 0.5

    problems = verify({"build": {"achieved_bpp": got}}, model_dir=None, required=[])
    assert any("contradicts the recipe's own serialized bytes" in p for p in problems)


def test_achieved_bpp_cross_check_passes_a_truthful_claim(tmp_path):
    """A claim slightly ABOVE the floor is expected, not a defect.

    The floor omits unpriced passthrough units, which only add bytes -- so the
    real DSv4 recipe claims 2.7555 against a 2.7385 floor (0.6% apart) and must
    pass. This pins the gate as a floor test, not an equality test.
    """
    from prismaquant.shipcard import allocator_achieved_bpp, verify

    recipe = _cb_recipe(
        tmp_path, units=4, bytes_per_unit=2654212, params_per_unit=8388608,
        meta={"achieved_bits": 2.53125 * 1.006},
    )
    got = allocator_achieved_bpp(recipe)
    assert got["cross_check"]["verdict"] == "agree"
    assert verify({"build": {"achieved_bpp": got}}, model_dir=None, required=[]) == []


def test_achieved_bpp_cross_check_is_silent_on_a_preexisting_card():
    """A card written before this gate carries no verdict and is left alone."""
    from prismaquant.shipcard import verify

    legacy = {"build": {"achieved_bpp": {"value": 4.3065, "source": "whatever"}}}
    assert verify(legacy, model_dir=None, required=[]) == []


def _add_passthrough(recipe, count):
    """Append FP8_SOURCE passthrough Linears, which carry no per-unit price."""
    payload = json.loads(recipe.read_text())
    for i in range(count):
        payload[f"model.layers.{i}.self_attn.kv_a_proj"] = {
            "bits": 8, "data_type": "fp8_e4m3",
        }
    recipe.write_text(json.dumps(payload))
    return recipe


def test_cross_check_does_not_refuse_a_passthrough_heavy_recipe(tmp_path):
    """A loose floor must not false-refuse a legal artifact at publication.

    Hy3 shipped 226 FP8_SOURCE Linears. Those carry no per-unit price, so they
    leave both sums -- and because passthrough is ~8.002 bpp against CB's ~2.5,
    the true rate sits legitimately far above the covered floor. Refusing that
    would be a gate bug that teaches the operator to reach for
    --force-unverified, which is a worse failure than not refusing.
    """
    from prismaquant.shipcard import allocator_achieved_bpp, verify

    recipe = _cb_recipe(
        tmp_path, units=4, bytes_per_unit=2654212, params_per_unit=8388608,
        meta={"achieved_bits": 5.2},        # honest for a half-passthrough mix
    )
    _add_passthrough(recipe, 4)             # coverage 4/8 = 50%

    cross = allocator_achieved_bpp(recipe)["cross_check"]
    assert cross["verdict"] == "inconclusive_low_coverage"
    assert cross["coverage_units"] == pytest.approx(0.5)
    assert cross["relative_difference"] > 1.0       # >100% apart, still not refused
    assert "too loose to indict" in cross["detail"]
    assert verify(
        {"build": {"achieved_bpp": allocator_achieved_bpp(recipe)}},
        model_dir=None, required=[],
    ) == []


def test_cross_check_refuses_a_claim_below_the_blend_at_high_coverage(tmp_path):
    """Undershoot is caught too, not just the stale-Pareto overshoot.

    At near-complete coverage the unpriced remainder cannot explain a gap in
    either direction, so a claim under the blend is as much a false public
    number as one over it.
    """
    from prismaquant.shipcard import allocator_achieved_bpp, verify

    recipe = _cb_recipe(
        tmp_path, units=100, bytes_per_unit=2654212, params_per_unit=8388608,
        meta={"achieved_bits": 1.5},        # well under the 2.53 blend
    )
    _add_passthrough(recipe, 2)             # coverage 100/102 = 98%

    got = allocator_achieved_bpp(recipe)
    cross = got["cross_check"]
    assert cross["verdict"] == "DISAGREE"
    assert cross["claim_is_below_floor"] is True
    assert cross["coverage_units"] > 0.95

    problems = verify({"build": {"achieved_bpp": got}}, model_dir=None, required=[])
    assert any("contradicts the recipe's own serialized bytes" in p for p in problems)


def test_cross_check_undershoot_is_also_coverage_gated(tmp_path):
    """The tempting shortcut -- "below the blend is impossible" -- is unsound.

    It holds only when every unpriced unit is HIGHER-rate than the priced
    blend. A recipe whose priced subset is FP8_CB-heavy (~8 bpp) with plain
    NVFP4 (4.25 bpp) unpriced has a true rate legitimately BELOW its own priced
    blend. The sound bound is blend x PARAMETER coverage, and parameter
    coverage is not computable from a recipe: non-CB units record no shape.
    So undershoot is gated on coverage exactly like overshoot.
    """
    from prismaquant.shipcard import allocator_achieved_bpp, verify

    recipe = _cb_recipe(
        tmp_path, units=4, bytes_per_unit=2654212, params_per_unit=8388608,
        meta={"achieved_bits": 1.5},
    )
    _add_passthrough(recipe, 4)             # coverage 50%

    got = allocator_achieved_bpp(recipe)
    cross = got["cross_check"]
    assert cross["verdict"] == "inconclusive_low_coverage"
    assert cross["claim_is_below_floor"] is True
    assert "below" in cross["detail"]
    assert verify({"build": {"achieved_bpp": got}}, model_dir=None, required=[]) == []


# ---------------------------------------------------------------------------
# native_export arm replay (issue #157)
# ---------------------------------------------------------------------------
def _native_record(slot, model_sha, metrics, *, passed=True):
    return make_record(
        slot=slot, tool="validate_native_export.py", passed=passed,
        model_sha=model_sha, metrics=metrics,
        detail=f"{slot} smoke", git_commit=_FAKE_COMMIT)


def test_native_export_refuses_a_mislabeled_arm(tmp_path):
    """An eager receipt must not close the graph slot (or vice versa).

    `validate_native_export._record_arm` stamps `metrics.arm` from the arm it
    actually ran, but `verify` compared only the slot key against
    `record["slot"]` — never `metrics.arm` against the slot suffix — so a
    mislabeled receipt verified clean.
    """
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    sha = compute_model_sha(model_dir)
    _fill_all(path, sha)
    fill_slot(path, "native_export.graph", _native_record(
        "native_export.graph", sha,
        {"arm": "eager", "generated_chars": 128, "enforce_eager": True,
         "max_new_tokens": 16}))

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any(
        p.startswith("native_export.graph:") and "metrics.arm" in p
        for p in problems
    ), problems


def test_native_export_refuses_a_pass_with_no_generation(tmp_path):
    """`passed=true` with `generated_chars=0` is not a smoke result.

    The eager/graph smoke's whole evidence is one greedy decode; a pass that
    generated nothing verified clean because `verify` never read the metrics.
    """
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    sha = compute_model_sha(model_dir)
    _fill_all(path, sha)
    fill_slot(path, "native_export.eager", _native_record(
        "native_export.eager", sha,
        {"arm": "eager", "generated_chars": 0, "enforce_eager": True,
         "max_new_tokens": 16}))

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any(
        p.startswith("native_export.eager:") and "generated_chars" in p
        for p in problems
    ), problems


def test_native_export_refuses_an_arm_running_under_the_wrong_residency(tmp_path):
    """`enforce_eager` must agree with the arm the slot names.

    On the Tessera lane the two residencies are different numeric objects
    exercised by separate gates; an eager-engine receipt sitting in the graph
    slot (or vice versa) is a mislabeled receipt, not a result.
    """
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    sha = compute_model_sha(model_dir)
    _fill_all(path, sha)
    fill_slot(path, "native_export.graph", _native_record(
        "native_export.graph", sha,
        {"arm": "graph", "generated_chars": 128, "enforce_eager": True,
         "max_new_tokens": 16}))

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any(
        p.startswith("native_export.graph:") and "enforce_eager" in p
        for p in problems
    ), problems


# ---------------------------------------------------------------------------
# build forensic replay (issue #158)
# ---------------------------------------------------------------------------
def _forensic_build(**overrides):
    """A build block shaped like the exporter's `_write_shipcard` output."""
    build = {
        "git": {"commit": _FAKE_COMMIT, "dirty": False},
        "source_model": "/models/Qwen3-4B",
        "layer_config": "/recipes/layer_config.json",
        "layer_config_sha": "b" * 64,
        "assignment_hash": "c" * 16,
        "config_assignment_hash": "d" * 16,
        "n_assignment_entries": 208,
        "achieved_bpp": {"value": 4.75},
        "read_gb_per_token": {"value": 3.127, "source": "measured"},
        "format_histogram": {"NVFP4/packed": 208},
        "render_levers": {"PRISMAQUANT_DO_NO_HARM": "1"},
        "kv_shared_fisher": kv_shared_fisher_echo({}),
    }
    build.update(overrides)
    return {"build": build}


def test_build_refuses_an_unvalidated_kv_fisher_correction():
    """An allocation that rode an under-counted h_trace must not verify.

    The exporter echoes the KV-cotangent / shared-Fisher flag state onto the
    card (D24), but `verify` never read it, so a card carrying
    `unvalidated_kv_fisher_correction=true` verified clean. The echo is
    derived from the function that owns it, not restated.
    """
    card = _forensic_build(kv_shared_fisher=kv_shared_fisher_echo(
        {"PRISMAQUANT_ALLOW_KV_SHARED_FISHER": "1"}))
    assert card["build"]["kv_shared_fisher"][
        "unvalidated_kv_fisher_correction"] is True

    problems = verify(card, model_dir=None, required=[])
    assert any("unvalidated_kv_fisher_correction" in p for p in problems), (
        problems)

    clean = _forensic_build()
    assert clean["build"]["kv_shared_fisher"][
        "unvalidated_kv_fisher_correction"] is False
    assert verify(clean, model_dir=None, required=[]) == []


def test_build_refuses_fabricated_forensic_hashes():
    """The stamped hashes must look like what the producer can stamp.

    `file_sha256` emits 64 hex chars or None; the assignment digests are 16
    hex chars or None. A card claiming any other string for them verified
    clean because nothing read the keys.
    """
    card = _forensic_build(layer_config_sha="whatever-the-recipe-was")
    problems = verify(card, model_dir=None, required=[])
    assert any("layer_config_sha" in p for p in problems), problems

    card = _forensic_build(assignment_hash="definitely-16-chars!!")
    problems = verify(card, model_dir=None, required=[])
    assert any("assignment_hash" in p for p in problems), problems


def test_build_refuses_a_malformed_format_histogram():
    """The histogram is a Counter rendering: string keys, positive counts."""
    card = _forensic_build(format_histogram={"NVFP4/packed": 0})
    problems = verify(card, model_dir=None, required=[])
    assert any("format_histogram" in p for p in problems), problems


def test_build_without_forensics_is_left_alone():
    """A card written before a forensic key existed carries no verdict.

    The replay fires only on keys the producer stamps — the same tolerance
    the `achieved_bpp` cross-check practices — so historical cards keep
    verifying.
    """
    assert verify({"build": {}}, model_dir=None, required=[]) == []
    legacy = {"build": {"achieved_bpp": {"value": 4.3065, "source": "x"}}}
    assert verify(legacy, model_dir=None, required=[]) == []
    assert verify(_forensic_build(), model_dir=None, required=[]) == []

