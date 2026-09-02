"""The `quantized_embedding` wire contract, producer side and across the seam.

The embedding is the one weight in the checkpoint that compressed-tensors
cannot carry: its vLLM path accepts weight-only INT schemes and RAISES for
FP8/NVFP4, so a stock config group naming the embedding does not mis-route the
artifact, it refuses to load it.  The declaration tested here is what lets
Gridbook own that unit instead.

The last test in this file is the one worth having: it hands the producer's
record to the CONSUMER's real parser, in the other repository, rather than to
the producer's re-implementation of it.  Two hand-maintained parsers agreeing
with each other is the failure mode a second implementation invites, and only
an end-to-end round trip can catch it.
"""
from __future__ import annotations

import pytest

from prismaquant.cb_export_config import (
    QUANTIZED_EMBEDDING_DECLARATION_KEY as KEY,
    QUANTIZED_EMBEDDING_DECLARATION_VERSION as VERSION,
    build_quantized_embedding_declaration,
    parse_quantized_embedding_declaration,
)


def test_builds_a_sorted_versioned_record():
    rec = build_quantized_embedding_declaration({"model.embed_tokens": "NVFP4"})
    assert rec == {"version": VERSION, "units": {"model.embed_tokens": "nvfp4"}}


def test_empty_units_refuses_rather_than_emitting_a_positive_claim():
    """Absence marks every artifact written before the key existed."""
    with pytest.raises(ValueError, match="must OMIT the key"):
        build_quantized_embedding_declaration({})


def test_lm_head_is_refused_at_write_time():
    """ParallelLMHead subclasses VocabParallelEmbedding.

    Declaring the head here dispatches the output projection as a LOOKUP
    instead of through compressed-tensors' linear method. The consumer refuses
    it; refusing here too turns a silent serving regression into a failed
    export.
    """
    with pytest.raises(ValueError, match="output projection"):
        build_quantized_embedding_declaration({"lm_head": "NVFP4"})
    with pytest.raises(ValueError, match="output projection"):
        build_quantized_embedding_declaration({"model.lm_head": "NVFP4"})


def test_an_unrouted_format_is_refused():
    """Adding a rung is a serving promotion, not a table entry."""
    with pytest.raises(ValueError, match="no consumer route"):
        build_quantized_embedding_declaration({"model.embed_tokens": "MXFP4"})


def test_absence_parses_as_no_units():
    assert parse_quantized_embedding_declaration({}) == {}


def test_unknown_version_refuses():
    with pytest.raises(ValueError, match="unsupported"):
        parse_quantized_embedding_declaration(
            {KEY: {"version": 99, "units": {"model.embed_tokens": "nvfp4"}}})


def test_a_unit_cannot_be_owned_by_two_dispatches():
    """Which dispatch wins would be decided by consumer branch order."""
    cfg = {
        KEY: {"version": VERSION, "units": {"model.embed_tokens": "nvfp4"}},
        "config_groups": {
            "g0": {"targets": ["model.embed_tokens"],
                   "weights": {"num_bits": 4}},
        },
    }
    with pytest.raises(ValueError, match="exactly one dispatch"):
        parse_quantized_embedding_declaration(cfg)


def test_producer_record_is_read_by_the_real_consumer_parser():
    """Cross-repo round trip: prismaquant writes it, gridbook reads it.

    ``gridbook.embedding`` imports vLLM lazily inside its methods, so the
    declaration half is importable in the build venv and this seam can be
    tested without a serving container.
    """
    gb = pytest.importorskip(
        "gridbook.embedding",
        reason="gridbook checkout not on sys.path; the seam is untested here")

    rec = build_quantized_embedding_declaration({"model.embed_tokens": "NVFP4"})
    units = gb.parse_declaration({gb.SCHEMA_KEY: rec},
                                 canonicalize=lambda n: n)

    assert set(units) == {"model.embed_tokens"}
    assert units["model.embed_tokens"] is gb.FORMATS["nvfp4"]
    # The two sides must agree on the KEY as well as the payload -- a producer
    # writing the right record under the wrong top-level name is a silent
    # no-op at load, which is worse than a refusal.
    assert KEY == gb.SCHEMA_KEY
    assert VERSION in gb.SUPPORTED_SCHEMA_VERSIONS


def test_every_producer_wire_id_is_routable_by_the_consumer():
    """A producer id the consumer does not know is a load failure."""
    gb = pytest.importorskip("gridbook.embedding")
    from prismaquant.cb_export_config import QUANTIZED_EMBEDDING_WIRE_IDS

    unroutable = set(QUANTIZED_EMBEDDING_WIRE_IDS.values()) - set(gb.FORMATS)
    assert not unroutable, (
        f"producer can emit wire ids the consumer cannot route: {unroutable}")
