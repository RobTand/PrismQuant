"""CPU refusal checks for the frozen qualification inputs and paired panels."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "experiments/pq237_streamed_protocol.py"
_SPEC = importlib.util.spec_from_file_location("pq237_streamed_protocol", _PATH)
protocol = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(protocol)


def _articles(n=32):
    rows = []
    for i in range(n):
        rows.extend([f" = Article {i} = \n", "\n", (f"Paragraph {i}. " * 100)])
    return rows


def _tokenizer(text, *, add_special_tokens):
    assert add_special_tokens is False
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return {"input_ids": [seed + i for i in range(70)]}


@pytest.fixture
def manifest(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (model / name).write_text(name)
    arrow = tmp_path / protocol.REVISION / "wikitext-validation.arrow"
    arrow.parent.mkdir()
    arrow.write_bytes(b"synthetic corpus byte identity; no real Arrow reads in this fixture")
    return protocol._make_protocol(model, arrow, _tokenizer, _articles())


def _write(tmp_path, manifest):
    raw = json.dumps(manifest).encode()
    path = tmp_path / "protocol.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_protocol_round_trip_and_portable_input_locations(tmp_path, manifest):
    path, digest = _write(tmp_path, manifest)
    assert protocol.load_protocol(path, digest) == manifest
    mapped = copy.deepcopy(manifest)
    mapped["model"]["path"] = "/unavailable/container/model"
    mapped["corpus"]["arrow_path"] = "/unavailable/container/corpus.arrow"
    path, digest = _write(tmp_path, mapped)
    assert protocol.load_protocol(path, digest, model_path=manifest["model"]["path"],
                                  corpus_arrow=manifest["corpus"]["arrow_path"]) == mapped


def test_changed_protocol_bytes_refuse_before_parse(tmp_path, manifest):
    path, digest = _write(tmp_path, manifest)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="immutable protocol digest"):
        protocol.load_protocol(path, digest)


@pytest.mark.parametrize("mutation,diagnostic", [
    (lambda m: m.update(n_probes=127), "n_probes"),
    (lambda m: m["plan"].pop(next(iter(protocol.PLAN))), "plan"),
    (lambda m: m["assignments"]["L0A8_L21A4"].update({"model.layers.0.mlp.down_proj": protocol.A16}), "assignments"),
    (lambda m: m["corpus"].update(revision="other"), "revision"),
    (lambda m: m["algorithm"].update(fresh_selection="choose_after_results"), "algorithm"),
    (lambda m: m["splits"]["fresh_holdout"].pop(), "exactly 32"),
    (lambda m: m["splits"]["fresh_holdout"][0].update(tokens=m["splits"]["original_calibration"][0]["tokens"]), "duplicate token sequence"),
    (lambda m: m["splits"]["fresh_holdout"][0].update(tokens=[True] * 64), "integer IDs"),
    (lambda m: m["splits"]["fresh_holdout"][0].update(sequence_id="wrong"), "sequence ID"),
    (lambda m: m["splits"]["original_holdout"][0].update(text="changed"), "paragraph hash"),
    (lambda m: m["splits"]["fresh_holdout"][1]["origin"].update(title="Article 0"), "origin/order"),
])
def test_protocol_structural_mutations_refuse_even_with_new_envelope_digest(tmp_path, manifest, mutation, diagnostic):
    mutation(manifest)
    path, digest = _write(tmp_path, manifest)
    with pytest.raises(ValueError, match=diagnostic):
        protocol.load_protocol(path, digest)


@pytest.mark.parametrize("kind", ["model", "tokenizer", "corpus", "extra_model_file"])
def test_changed_input_files_refuse(tmp_path, manifest, kind):
    path, digest = _write(tmp_path, manifest)
    if kind == "corpus":
        changed = Path(manifest["corpus"]["arrow_path"])
    else:
        changed = Path(manifest["model"]["path"]) / {
            "model": "model.safetensors", "tokenizer": "tokenizer.json", "extra_model_file": "added.json"}[kind]
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch|file identity mismatch"):
        protocol.load_protocol(path, digest)


def test_selection_uses_first_qualifying_paragraph_and_distinct_top_level_articles():
    rows = [" = Too short = ", "short", "later long " * 150,
            " = Duplicate = ", "short", " = Duplicate = ", "ignored " * 150,
            " = Has subsection first = ", " == Section == ", "ignored " * 150,
            *_articles()]
    selected = protocol._select_articles(rows)
    assert len(selected) == 32
    assert [record["origin"]["title"] for record in selected] == [
        "Too short", "Has subsection first", *[f"Article {i}" for i in range(30)]]
    assert selected[0]["origin"]["paragraph_row"] == 2
    with pytest.raises(ValueError, match="fewer than 32"):
        protocol._select_articles(_articles(31))


def test_short_intro_does_not_exclude_an_otherwise_qualifying_article():
    rows = []
    for i in range(32):
        rows.extend([f" = Article {i} = ", "short introduction", " == Section == ",
                     f"First long {i}. " * 100, f"Second long {i}. " * 100])
    selected = protocol._select_articles(rows)
    assert len(selected) == 32
    assert [record["origin"]["paragraph_row"] for record in selected] == [i * 5 + 3 for i in range(32)]
    assert all(record["text"].startswith("First long") for record in selected)


def test_paired_summary_retains_cancellation_and_descriptive_scope():
    actual = protocol.paired_sequence_summary({"a": 4.0, "b": 1.0, "c": 9.0},
                                               {"a": 2.0, "b": 3.0, "c": 9.0})
    assert actual["difference_per_sequence"] == [2.0, -2.0, 0.0]
    assert actual["mean_difference"] == 0
    assert actual["descriptive_sequence_standard_error"] == pytest.approx(2 / math.sqrt(3))
    assert actual["scope"] == "descriptive_panel_only"
    assert "confidence_interval" not in actual


@pytest.mark.parametrize("left,right", [
    ({"a": 1.0}, {"a": 2.0}),
    ({"a": 1.0, "b": 2.0}, {"b": 2.0, "a": 1.0}),
    ({"a": float("nan"), "b": 2.0}, {"a": 1.0, "b": 2.0}),
    ({"a": True, "b": 2.0}, {"a": 1.0, "b": 2.0}),
    ({"a": 1e308, "b": 2.0}, {"a": -1e308, "b": 2.0}),
])
def test_paired_summary_refuses_unaligned_or_invalid_samples(left, right):
    with pytest.raises(ValueError):
        protocol.paired_sequence_summary(left, right)
