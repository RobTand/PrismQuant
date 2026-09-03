"""The byte-matched uniform control gate (#121, closing #117 / tessera#1).

An allocation over a rate axis is a claim: that choosing a rung per Linear
beats spending the same bytes everywhere.  Measured and served 2026-09-02, on
the artifact these numbers come from, that claim was FALSE by 2.00x while
every other check the pipeline owns passed.  These tests pin the refusal, the
one deliberate way past it, and the four ways the gate can be lied to.

The control block below is the 2026-09-02 receipt's own, verbatim from
RobTand/prismaquant#121 -- so the fixtures are a measurement, not a mock.
"""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

if not (pathlib.Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

from prismaquant.shipcard import (
    GOLD_SLOTS,
    REQUIRED_SLOTS,
    UNIFORM_CONTROL_SLOT,
    build_shipcard,
    compute_model_sha,
    fill_slot,
    load_shipcard,
    make_record,
    make_uniform_control_record,
    required_slots,
    uniform_control_summary,
    verify,
    write_shipcard,
)
from prismaquant.shipcard_cli import main as shipcard_cli
from prismaquant.validate_quantized_model import (
    DEFAULT_MAX_MEAN_NLL,
    DEFAULT_MAX_P99_NLL,
    DEFAULT_MAX_PPL,
    DEFAULT_MIN_GEN_LEN,
    DEFAULT_MIN_MTP_ACCEPT_P0,
)
from tools.publish_artifact import check_shipcard, main as publish_cli

_FAKE_FINGERPRINT = "f" * 64
_CONTROL_FINGERPRINT = "e" * 64
_FAKE_COMMIT = "a" * 40
_CONTROL_SHA = "b" * 64

#: The receipt's own arms: allocated 0.3485 against the byte-matched uniform
#: 0.1746, 1.9959908361970216x, control 65.1 ppm the fatter arm.
_ALLOCATED_KL = 0.3485
_CONTROL_KL = 0.1746

_CONTROL_BLOCK = {
    "schema": "tessera.uniform_control.v1",
    "candidate_label": "allocated",
    "control": {
        "grid": "E4M3", "q256": 1006, "rule": "nearest",
        "searched_q256": [256, 2048], "legal_rungs": 1793,
        "bracket": {"below": {"q256": 1005, "bits": 1760116736},
                    "above": {"q256": 1006, "bits": 1761837056},
                    "quantum_bits": 1720320},
        "units": 196, "bf16_carried": 0,
        "dominated_by": None,
        "match": {
            "candidate_bits": 1761722368, "control_bits": 1761837056,
            "varying_params": 440401920,
            "candidate_bpp": 4.000260416666666,
            "control_bpp": 4.000520833333334,
            "slack_bits": 114688, "relative_slack_ppm": 65.09992839007877,
            "fatter_arm": "control", "control_is_no_larger": False,
            "max_relative_slack": [1, 1000], "byte_matched": True,
        },
    },
    "verdict": {
        "metric": "kl_vs_bf16", "measured": True,
        "candidate": _ALLOCATED_KL, "control": _CONTROL_KL,
        "candidate_over_control": _ALLOCATED_KL / _CONTROL_KL,
        "beat_control": False,
    },
}


def _block(*, candidate=_ALLOCATED_KL, control=_CONTROL_KL, measured=True):
    block = copy.deepcopy(_CONTROL_BLOCK)
    if not measured:
        block["verdict"] = {
            "metric": "kl_vs_bf16", "measured": False,
            "detail": "the control was built and priced; neither arm was served",
        }
        return block
    block["verdict"].update({
        "candidate": candidate,
        "control": control,
        "candidate_over_control": candidate / control,
        "beat_control": candidate < control,
    })
    return block


def _gold_metrics(candidate_kl):
    return {
        "gold.kl": {
            "kl_mean": candidate_kl,
            "kl_confident_mean": candidate_kl * 0.95,
            "n_positions": 4088,
            "n_samples": 8,
            "seqlen": 512,
            "score_positions": "all",
        },
        "gold.ppl": {"ppl": 8.33, "mean_nll": 2.12, "n_tokens_scored": 8192},
    }


def _control_arm(control_kl=_CONTROL_KL, **overrides):
    arm = {
        "tool": "test",
        "model_sha": _CONTROL_SHA,
        "git_commit": _FAKE_COMMIT,
        "serve_fingerprint": _CONTROL_FINGERPRINT,
        "spec_decode_detected": False,
        "metrics": {
            "kl_mean": control_kl,
            "kl_confident_mean": control_kl * 0.95,
            "n_positions": 4088,
            "n_samples": 8,
            "seqlen": 512,
            "score_positions": "all",
        },
    }
    metrics = overrides.pop("metrics", None)
    arm.update(overrides)
    if metrics:
        arm["metrics"] = {**arm["metrics"], **metrics}
    return arm


def _artifact(tmp_path, *, name="exported", rate_axis=True):
    model_dir = tmp_path / name
    model_dir.mkdir()
    config = {"model_type": "qwen3"}
    if rate_axis:
        config["quantization_config"] = {
            "quant_method": "tessera", "format": "mixed-precision"}
    (model_dir / "config.json").write_text(json.dumps(config))
    # Distinct bytes per artifact: two directories with identical bytes ARE
    # one checkpoint, and the gate says so ("compared against itself").
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(
        b"weights of " + name.encode())
    card = build_shipcard(model_dir, build={"achieved_bpp": {"value": 4.0}})
    write_shipcard(model_dir / "shipcard.json", card)
    return model_dir


#: What a real ship-gate verdict carries. Thresholds come from the
#: producer's own DEFAULT_* constants; the four ledger names are the checks
#: `run_validation` files (the ledger's key set is pinned to the producer,
#: not restated, by tests/test_shipcard.py::_producer_check_names).
_SHIP_GATE_LEDGER = ("serve_ready", "generation_sanity", "perplexity",
                     "mtp_acceptance")


def _ship_gate_record(model_sha, *, source):
    ledger = {name: {"passed": True} for name in _SHIP_GATE_LEDGER}
    ledger["perplexity"] = {
        "passed": True,
        "perplexity": 8.33,
        "mean_nll_per_tok": 2.12,
        "max_nll_per_tok": 4.50,
        "n_tokens": 8192,
        "spec_decode_detected": False,
    }
    return make_record(
        slot="ship_gate", tool="validate_quantized_model.py", passed=True,
        model_sha=model_sha, metrics=ledger,
        detail="serve_ready=pass; generation_sanity=pass; "
               "perplexity=pass; mtp_acceptance=pass",
        spec_decode_detected=False,
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
                "bos_token": None,
                "add_special_tokens": True,
            },
            "model_sha_source": source,
        },
    )


def _native_record(slot, model_sha):
    # What `validate_native_export._record_arm` files: the arm it ran
    # (`arm = "eager" if enforce_eager else "graph"`) with one greedy
    # decode as evidence.
    arm = slot.split(".", 1)[1]
    return make_record(
        slot=slot, tool="validate_native_export.py", passed=True,
        model_sha=model_sha,
        metrics={"arm": arm, "generated_chars": 128,
                 "enforce_eager": arm == "eager", "max_new_tokens": 16},
        detail=f"{arm} smoke", git_commit=_FAKE_COMMIT)


def _close_base_slots(model_dir, candidate_kl=_ALLOCATED_KL):
    path = model_dir / "shipcard.json"
    sha = compute_model_sha(model_dir)
    metrics = _gold_metrics(candidate_kl)
    for slot in REQUIRED_SLOTS:
        if slot == "ship_gate":
            fill_slot(path, slot, _ship_gate_record(sha, source=str(model_dir)))
            continue
        if slot.startswith("native_export."):
            fill_slot(path, slot, _native_record(slot, sha))
            continue
        is_gold = slot in GOLD_SLOTS
        fill_slot(path, slot, make_record(
            slot=slot,
            tool="test",
            passed=True,
            model_sha=sha,
            spec_decode_detected=(False if is_gold else None),
            metrics=(metrics.get(slot) if is_gold else None),
            serve_fingerprint=(_FAKE_FINGERPRINT if is_gold else None),
            git_commit=(_FAKE_COMMIT if is_gold else None),
        ))
    return path


def _fill_control(path, model_dir, *, block, arm=None, key="kl_mean"):
    record = make_uniform_control_record(
        tool="test",
        model_sha=compute_model_sha(model_dir),
        control_block=block,
        control_arm=arm if arm is not None else _control_arm(),
        gold_metric_key=key,
        git_commit=_FAKE_COMMIT,
    )
    card = load_shipcard(path)
    card["slots"][UNIFORM_CONTROL_SLOT] = record
    write_shipcard(path, card)
    return record


def _built(tmp_path, *, candidate=_ALLOCATED_KL, control=_CONTROL_KL,
           measured=True, arm=None, rate_axis=True, key="kl_mean"):
    """An artifact with every base slot closed and one control verdict."""
    model_dir = _artifact(tmp_path, rate_axis=rate_axis)
    path = _close_base_slots(model_dir, candidate_kl=candidate)
    _fill_control(
        path, model_dir,
        block=_block(candidate=candidate, control=control, measured=measured),
        arm=arm if arm is not None else _control_arm(control),
        key=key,
    )
    return model_dir, path


def _problems(model_dir, path):
    return verify(load_shipcard(path), model_dir=model_dir)


def _publish(model_dir, *extra):
    return publish_cli([
        str(model_dir), "--repo-id", "rdtand/test-artifact", "--dry-run",
        *extra,
    ])


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def test_an_allocation_that_beats_its_control_is_publishable(tmp_path):
    model_dir, path = _built(tmp_path, candidate=0.1500)

    assert _problems(model_dir, path) == []
    assert check_shipcard(model_dir, path)[1] == []
    assert _publish(model_dir) == 0

    summary = uniform_control_summary(load_shipcard(path), model_dir=model_dir)
    assert summary["beat_control"] is True
    assert summary["overridden"] is False


def test_a_losing_allocation_cannot_be_published(tmp_path):
    """The receipt's own arms: 0.3485 against 0.1746 at matched bytes."""
    model_dir, path = _built(tmp_path)

    problems = _problems(model_dir, path)
    assert any("LOST to its byte-matched uniform control" in p
               for p in problems), problems
    assert any("1.996x worse" in p for p in problems), problems

    # Not merely a field: publication itself refuses, and says so.
    assert check_shipcard(model_dir, path)[1] != []
    assert _publish(model_dir) == 1
    # ... and the refusal is forceable evidence, not a malformed card.
    assert check_shipcard(model_dir, path)[0] is not None


def test_a_hand_set_passed_flag_does_not_admit_a_losing_allocation(tmp_path):
    model_dir, path = _built(tmp_path)
    card = load_shipcard(path)
    card["slots"][UNIFORM_CONTROL_SLOT]["passed"] = True
    write_shipcard(path, card)

    problems = _problems(model_dir, path)
    assert any("LOST to its byte-matched uniform control" in p
               for p in problems), problems
    assert any("passed=True while its own KLs say" in p
               for p in problems), problems
    assert _publish(model_dir) == 1


def test_a_control_that_was_never_served_is_not_a_pass(tmp_path):
    model_dir, path = _built(tmp_path, measured=False)

    problems = _problems(model_dir, path)
    assert any("built and priced but never SERVED" in p
               for p in problems), problems
    assert any("Missing is not passing" in p for p in problems), problems
    assert _publish(model_dir) == 1


def test_a_rate_axis_artifact_with_no_control_refuses_rather_than_passing(
    tmp_path,
):
    model_dir = _artifact(tmp_path)
    path = _close_base_slots(model_dir)
    card = load_shipcard(path)
    card["slots"][UNIFORM_CONTROL_SLOT] = None
    write_shipcard(path, card)

    assert UNIFORM_CONTROL_SLOT in required_slots(card, model_dir=model_dir)
    assert f"{UNIFORM_CONTROL_SLOT}: UNFILLED" in _problems(model_dir, path)
    assert _publish(model_dir) == 1
    summary = uniform_control_summary(card, model_dir=model_dir)
    assert summary["applicable"] is True and summary["filled"] is False
    assert "NOT MEASURED" in summary["detail"]


def test_the_obligation_survives_erasing_either_half_of_its_evidence(tmp_path):
    """Nulling the card's build block or losing the config does not erase it."""
    model_dir = _artifact(tmp_path)
    card = load_shipcard(_close_base_slots(model_dir))

    card["build"] = {}
    assert UNIFORM_CONTROL_SLOT in required_slots(card, model_dir=model_dir)

    # ... and from the card alone, with no artifact to read.
    card["build"] = {"export_container": "tessera"}
    assert UNIFORM_CONTROL_SLOT in required_slots(card, model_dir=None)


def test_a_format_menu_artifact_owes_no_uniform_control(tmp_path):
    """No rung axis, no uniform rung: a gate no correct artifact can pass."""
    model_dir, path = _built(tmp_path, rate_axis=False, candidate=0.1500)
    card = load_shipcard(path)
    card["slots"].pop(UNIFORM_CONTROL_SLOT)
    write_shipcard(path, card)
    card = load_shipcard(path)

    assert required_slots(card, model_dir=model_dir) == REQUIRED_SLOTS
    assert _problems(model_dir, path) == []
    summary = uniform_control_summary(card, model_dir=model_dir)
    assert summary["applicable"] is False
    assert "no rate axis" in summary["detail"]


# ---------------------------------------------------------------------------
# The four ways to lie to it
# ---------------------------------------------------------------------------
def test_arms_that_are_not_byte_matched_refuse(tmp_path):
    block = _block(candidate=0.1500)
    block["control"]["match"]["control_bits"] = 1800000000
    block["control"]["match"]["slack_bits"] = 1800000000 - 1761722368
    block["control"]["match"]["control_bpp"] = 1800000000 / 440401920
    model_dir = _artifact(tmp_path)
    path = _close_base_slots(model_dir, candidate_kl=0.1500)
    _fill_control(path, model_dir, block=block)

    problems = _problems(model_dir, path)
    assert any("NOT byte-matched" in p for p in problems), problems
    assert any("byte_matched=True but its own integers replay to False" in p
               for p in problems), problems
    assert _publish(model_dir) == 1


def test_a_block_cannot_widen_its_own_tolerance(tmp_path):
    block = _block(candidate=0.1500)
    match = block["control"]["match"]
    match["max_relative_slack"] = [1, 10]
    match["control_bits"] = 1780000000
    match["slack_bits"] = 1780000000 - 1761722368
    match["control_bpp"] = 1780000000 / 440401920
    model_dir = _artifact(tmp_path)
    path = _close_base_slots(model_dir, candidate_kl=0.1500)
    _fill_control(path, model_dir, block=block)

    problems = _problems(model_dir, path)
    assert any("widened its own tolerance" in p for p in problems), problems
    assert _publish(model_dir) == 1


def test_a_last_token_screen_cannot_close_this_slot(tmp_path):
    """The control arm replays the gold contract: `final` is triage only."""
    arm = _control_arm()
    arm["metrics"]["score_positions"] = "final"
    model_dir, path = _built(tmp_path, candidate=0.1500, arm=arm)

    problems = _problems(model_dir, path)
    assert any("last-token hook screen" in p for p in problems), problems
    assert _publish(model_dir) == 1


def test_the_candidate_arm_must_be_the_cards_own_gold_kl(tmp_path):
    """A block measured on some other allocation cannot be pasted on."""
    model_dir = _artifact(tmp_path)
    path = _close_base_slots(model_dir, candidate_kl=0.0151)
    _fill_control(path, model_dir, block=_block(candidate=0.1500))

    problems = _problems(model_dir, path)
    assert any("is not the card's own gold.kl" in p for p in problems), problems
    assert _publish(model_dir) == 1


def test_the_two_arms_must_share_one_measurement_contract(tmp_path):
    arm = _control_arm(metrics={"n_samples": 4, "n_positions": 2044})
    model_dir, path = _built(tmp_path, candidate=0.1500, arm=arm)

    problems = _problems(model_dir, path)
    assert any("did not run the same measurement contract" in p
               for p in problems), problems
    assert _publish(model_dir) == 1


def test_the_control_must_be_a_second_checkpoint(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _close_base_slots(model_dir, candidate_kl=0.1500)
    arm = _control_arm(model_sha=compute_model_sha(model_dir))
    _fill_control(path, model_dir, block=_block(candidate=0.1500), arm=arm)

    problems = _problems(model_dir, path)
    assert any("compared against itself is not a control" in p
               for p in problems), problems


def test_a_dominated_control_is_a_handicapped_arm(tmp_path):
    block = _block(candidate=0.1500)
    block["control"]["dominated_by"] = 1010
    model_dir = _artifact(tmp_path)
    path = _close_base_slots(model_dir, candidate_kl=0.1500)
    _fill_control(path, model_dir, block=block)

    problems = _problems(model_dir, path)
    assert any("handicapped uniform arm" in p for p in problems), problems


def test_a_spec_decode_tainted_control_arm_refuses(tmp_path):
    arm = _control_arm(spec_decode_detected=True)
    model_dir, path = _built(tmp_path, candidate=0.1500, arm=arm)

    problems = _problems(model_dir, path)
    assert any("spec_decode_detected is TRUE" in p for p in problems), problems


def test_a_verdict_that_disagrees_with_its_own_two_kls_refuses(tmp_path):
    block = _block(candidate=0.1500)
    block["verdict"]["candidate_over_control"] = 0.5
    model_dir = _artifact(tmp_path)
    path = _close_base_slots(model_dir, candidate_kl=0.1500)
    _fill_control(path, model_dir, block=block)

    problems = _problems(model_dir, path)
    assert any("candidate_over_control=0.5 but" in p for p in problems), problems


def test_the_gate_is_on_kl_vs_bf16_and_nothing_else(tmp_path):
    block = _block(candidate=0.1500)
    block["verdict"]["metric"] = "weight_mse"
    model_dir = _artifact(tmp_path)
    path = _close_base_slots(model_dir, candidate_kl=0.1500)
    _fill_control(path, model_dir, block=block)

    problems = _problems(model_dir, path)
    assert any("this gate is on the serving metric" in p
               for p in problems), problems


# ---------------------------------------------------------------------------
# The override
# ---------------------------------------------------------------------------
def test_the_override_admits_a_measured_loss_and_stamps_the_card(tmp_path):
    model_dir, path = _built(tmp_path)
    assert _publish(model_dir) == 1

    assert shipcard_cli([
        "override-control", str(path),
        "--reason", "shipping the allocated arm to reproduce the receipt",
        "--authorized-by", "robert.tand@icloud.com",
        "--confirm-name", model_dir.name,
    ]) == 0

    card = load_shipcard(path)
    assert card["uniform_control_override"] is True
    history = card["uniform_control_override_history"]
    assert len(history) == 1
    assert history[0]["confirmed_artifact_name"] == model_dir.name
    assert history[0]["model_sha"] == card["model_sha"]
    assert history[0]["candidate_over_control"] == pytest.approx(
        _ALLOCATED_KL / _CONTROL_KL)

    assert _problems(model_dir, path) == []
    assert _publish(model_dir) == 0
    assert uniform_control_summary(card, model_dir=model_dir)["overridden"]


def test_the_override_requires_the_basename_retyped(tmp_path):
    model_dir, path = _built(tmp_path)

    assert shipcard_cli([
        "override-control", str(path),
        "--reason", "r", "--authorized-by", "a",
        "--confirm-name", "not-the-artifact",
    ]) == 1
    assert "uniform_control_override" not in load_shipcard(path)
    assert _publish(model_dir) == 1


def test_the_override_does_not_survive_a_remeasurement(tmp_path):
    model_dir, path = _built(tmp_path)
    shipcard_cli([
        "override-control", str(path),
        "--reason", "r", "--authorized-by", "a",
        "--confirm-name", model_dir.name,
    ])
    assert _problems(model_dir, path) == []

    # Re-serve both arms; the loss is now a different loss.
    card = load_shipcard(path)
    record = card["slots"][UNIFORM_CONTROL_SLOT]
    record["uniform_control"] = _block(candidate=0.3000)
    record["control_arm"] = _control_arm()
    for slot in ("gold.kl",):
        card["slots"][slot]["metrics"]["kl_mean"] = 0.3000
        card["slots"][slot]["metrics"]["kl_confident_mean"] = 0.3000 * 0.95
    write_shipcard(path, card)

    problems = _problems(model_dir, path)
    assert any("does not survive a re-measurement" in p
               for p in problems), problems
    assert _publish(model_dir) == 1


def test_the_override_cannot_be_used_to_skip_the_measurement(tmp_path):
    model_dir, path = _built(tmp_path, measured=False)

    assert shipcard_cli([
        "override-control", str(path),
        "--reason", "r", "--authorized-by", "a",
        "--confirm-name", model_dir.name,
    ]) == 2
    assert "uniform_control_override" not in load_shipcard(path)
    assert _publish(model_dir) == 1


def test_an_override_pasted_onto_an_unserved_record_still_refuses(tmp_path):
    """Bypassing the CLI does not bypass the rule it enforces."""
    model_dir, path = _built(tmp_path, measured=False)
    card = load_shipcard(path)
    card["slots"][UNIFORM_CONTROL_SLOT]["override"] = {
        "schema": "prismaquant.uniform_control_override/1",
        "reason": "r", "authorized_by": "a",
        "confirmed_artifact_name": model_dir.name,
        "model_sha": card["model_sha"],
        "candidate_over_control": _ALLOCATED_KL / _CONTROL_KL,
        "stamped_at": "2026-09-03T00:00:00Z",
    }
    write_shipcard(path, card)

    problems = _problems(model_dir, path)
    assert any("never SERVED" in p for p in problems), problems
    assert _publish(model_dir) == 1


def test_an_override_does_not_forgive_a_control_that_is_not_byte_matched(
    tmp_path,
):
    block = _block()
    block["control"]["match"]["control_bits"] = 1800000000
    block["control"]["match"]["slack_bits"] = 1800000000 - 1761722368
    block["control"]["match"]["control_bpp"] = 1800000000 / 440401920
    model_dir = _artifact(tmp_path)
    path = _close_base_slots(model_dir)
    _fill_control(path, model_dir, block=block)
    card = load_shipcard(path)
    card["slots"][UNIFORM_CONTROL_SLOT]["override"] = {
        "schema": "prismaquant.uniform_control_override/1",
        "reason": "r", "authorized_by": "a",
        "confirmed_artifact_name": model_dir.name,
        "model_sha": card["model_sha"],
        "candidate_over_control": _ALLOCATED_KL / _CONTROL_KL,
        "stamped_at": "2026-09-03T00:00:00Z",
    }
    write_shipcard(path, card)

    problems = _problems(model_dir, path)
    assert any("The override on this record does not apply" in p
               for p in problems), problems
    assert _publish(model_dir) == 1


def test_the_override_refuses_when_there_is_nothing_to_override(tmp_path):
    model_dir, path = _built(tmp_path, candidate=0.1500)
    assert shipcard_cli([
        "override-control", str(path),
        "--reason", "r", "--authorized-by", "a",
        "--confirm-name", model_dir.name,
    ]) == 2


# ---------------------------------------------------------------------------
# The fill path
# ---------------------------------------------------------------------------
def test_the_override_survives_the_directory_being_renamed(tmp_path):
    """The re-typed basename is a stamp-time ceremony, not a verify-time key.

    ``--force-unverified`` checks the re-typed name against the directory at
    the moment of typing and never again; the override must do the same,
    because the publisher verifies a *snapshot copy* under a randomised name
    and a downloaded artifact sits wherever the downloader put it.  What the
    override binds to at verify time is the card: model_sha and the forgiven
    ratio.  Before this test, publishing any overridden artifact refused with
    "override confirms 'exported' but the artifact directory is
    '.prismaquant-publish-snapshot-…'".
    """
    model_dir, path = _built(tmp_path)
    assert shipcard_cli([
        "override-control", str(path),
        "--reason", "r", "--authorized-by", "a",
        "--confirm-name", model_dir.name,
    ]) == 0
    moved = tmp_path / "somewhere-else"
    model_dir.rename(moved)
    assert _problems(moved, moved / "shipcard.json") == []
    assert _publish(moved) == 0


def test_fill_control_closes_the_slot_from_two_measurements(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _close_base_slots(model_dir, candidate_kl=0.1500)
    control_dir = _artifact(tmp_path, name="control")

    block_path = tmp_path / "control_block.json"
    block_path.write_text(json.dumps(_block(candidate=0.1500)))
    record_path = tmp_path / "control_kl.json"
    record_path.write_text(json.dumps({
        "model": str(control_dir),
        "kl_mean": _CONTROL_KL,
        "kl_confident_mean": _CONTROL_KL * 0.95,
        "n_positions": 4088, "n_samples": 8, "seqlen": 512,
        "score_positions": "all",
        "serve_fingerprint": _CONTROL_FINGERPRINT,
        "git_commit": _FAKE_COMMIT,
        "spec_decode_detected": False,
    }))

    assert shipcard_cli([
        "fill-control", str(path),
        "--control-block", str(block_path),
        "--control-record", str(record_path),
        "--tool", "test",
    ]) == 0
    assert _problems(model_dir, path) == []
    assert _publish(model_dir) == 0


def test_fill_control_refuses_an_unserved_block_without_the_flag(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _close_base_slots(model_dir)
    control_dir = _artifact(tmp_path, name="control")
    block_path = tmp_path / "control_block.json"
    block_path.write_text(json.dumps(_block(measured=False)))
    record_path = tmp_path / "control_kl.json"
    record_path.write_text(json.dumps({"model": str(control_dir)}))

    assert shipcard_cli([
        "fill-control", str(path),
        "--control-block", str(block_path),
        "--control-record", str(record_path),
    ]) == 2
    assert load_shipcard(path)["slots"].get(UNIFORM_CONTROL_SLOT) is None
