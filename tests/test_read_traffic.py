"""Anchors for the per-token decode read-bytes stat.

The load-bearing test is :func:`test_synthetic_ledger_matches_hand_computation`:
every byte in a tiny synthetic checkpoint is written out by hand below, and the
module must reproduce the ledger and the weighted total exactly.  A stat that
is "about right" is worthless as a bandwidth ceiling, so the anchor is exact
integers, not tolerances.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from prismaquant import footprint as fp
from prismaquant import name_projection as npx
from prismaquant import read_traffic as rt
from prismaquant.model_profiles import detect_profile_with_warning

# ---------------------------------------------------------------------------
# The synthetic model, written out by hand.
#
# Everything is BF16 on disk (2 bytes/param), one body layer, 4 routed experts
# of which a token activates 2.
#
#   name                                              shape   params  bytes
#   model.embed_tokens.weight                        (16, 8)     128    256
#   model.layers.0.self_attn.q_proj.weight            (8, 8)      64    128
#   model.layers.0.mlp.gate.weight            (router)(4, 8)      32     64
#   model.layers.0.mlp.experts.{0..3}.gate_proj.w..   (4, 8)   4x 32  4x 64
#   model.layers.0.input_layernorm.weight               (8,)       8     16
#   lm_head.weight                                   (16, 8)     128    256
#   mtp.fc.weight                             (draft) (8, 8)      64    128
#                                                          source total 1104
# ---------------------------------------------------------------------------
HIDDEN, INTER, VOCAB, N_EXPERTS, TOPK = 8, 4, 16, 4, 2

_TENSORS: dict[str, tuple[int, ...]] = {
    "model.embed_tokens.weight": (VOCAB, HIDDEN),
    "model.layers.0.self_attn.q_proj.weight": (HIDDEN, HIDDEN),
    "model.layers.0.mlp.gate.weight": (N_EXPERTS, HIDDEN),
    **{
        f"model.layers.0.mlp.experts.{i}.gate_proj.weight": (INTER, HIDDEN)
        for i in range(N_EXPERTS)
    },
    "model.layers.0.input_layernorm.weight": (HIDDEN,),
    "lm_head.weight": (VOCAB, HIDDEN),
    "mtp.fc.weight": (HIDDEN, HIDDEN),
}

SOURCE_TOTAL_BYTES = 1104

# The two format numbers the ledger depends on, stated so the arithmetic
# below is checkable without running the registry.  FP8_E4M3 stores one byte
# per parameter plus an fp32 scale plane:
#   (8, 8)    -> 64 weight bytes + 8 row scales x 4 B  =  96 B
#   (4, 4, 8) -> 128 weight bytes + 4 expert scales x 4 B = 144 B
FP8_Q_PROJ_BYTES = 96
FP8_EXPERT_STACK_BYTES = 144

PACKED_EXPERTS = "model.layers.0.mlp.experts.gate_up_proj"
Q_PROJ = "model.layers.0.self_attn.q_proj"


_ITEMSIZE = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1, "I64": 8}


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[int, ...]],
    dtypes: dict[str, str] | None = None,
) -> None:
    """Write a valid safetensors shard; only the header is ever read."""
    header: dict[str, object] = {}
    offset = 0
    for name, shape in tensors.items():
        n = 1
        for dim in shape:
            n *= dim
        dtype = (dtypes or {}).get(name, "BF16")
        nbytes = n * _ITEMSIZE[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic-moe"
    root.mkdir()
    _write_safetensors(root / "model.safetensors", _TENSORS)
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "num_experts": N_EXPERTS,
        "num_experts_per_tok": TOPK,
    }))
    return root


@pytest.fixture()
def profile(model_dir: Path):
    return detect_profile_with_warning(str(model_dir), entrypoint="test")


ASSIGNMENT = {Q_PROJ: "FP8_DYNAMIC", PACKED_EXPERTS: "FP8_DYNAMIC"}
STATS = {
    Q_PROJ: {"out_features": HIDDEN, "in_features": HIDDEN,
             "n_params": HIDDEN * HIDDEN},
    PACKED_EXPERTS: {"num_experts": N_EXPERTS, "out_features": INTER,
                     "in_features": HIDDEN,
                     "n_params": N_EXPERTS * INTER * HIDDEN},
}


def _report(model_dir: Path, profile, **kwargs) -> dict:
    return rt.assignment_read_traffic(
        ASSIGNMENT, STATS, model_path=str(model_dir), profile=profile,
        **kwargs)


def test_source_partition_covers_every_checkpoint_byte(model_dir: Path):
    """The floor half is only honest if it provably misses nothing."""
    spans = fp.source_tensor_span_bytes(str(model_dir))
    assert sum(spans.values()) == SOURCE_TOTAL_BYTES
    assert sum(spans.values()) == fp.source_checkpoint_bytes(str(model_dir))[0]
    assert set(spans) == set(_TENSORS)


def test_synthetic_ledger_matches_hand_computation(model_dir: Path, profile):
    report = _report(model_dir, profile)

    # --- stored bytes, class by class -------------------------------------
    # dense           = q_proj re-encoded to FP8                       96
    # routed_experts  = the 4-expert stack re-encoded to FP8          144
    # held_fixed      = router 64 + layernorm 16 + lm_head 256        336
    # excluded_embed  = embed_tokens                                  256
    # excluded_mtp    = the draft sidecar                             128
    #                                                          total  960
    classes = report["classes"]
    assert classes["dense"]["stored_bytes"] == FP8_Q_PROJ_BYTES
    assert classes["routed_experts"]["stored_bytes"] == FP8_EXPERT_STACK_BYTES
    assert classes["held_fixed"]["stored_bytes"] == 64 + 16 + 256
    assert classes["excluded_embedding"]["stored_bytes"] == 256
    assert classes["excluded_mtp"]["stored_bytes"] == 128
    assert classes["excluded_non_text_graph"]["stored_bytes"] == 0
    assert classes["resident_codebooks"]["stored_bytes"] == 0

    # --- the reconciliation the module refuses to run without --------------
    priced = fp.assignment_artifact_bytes(
        ASSIGNMENT, STATS,
        source_total_bytes=SOURCE_TOTAL_BYTES,
        source_manifest=fp.source_tensor_bytes_manifest(
            str(model_dir), profile.checkpoint_to_live_name,
            profile.packed_expert_parent_for_projection),
        cb_serialization_context=None,
    )
    assert priced["artifact_payload_bytes"] == 960
    assert report["reconciliation"]["ledger_stored_bytes"] == 960
    assert report["reconciliation"]["footprint_artifact_payload_bytes"] == 960

    # --- the weighted total ------------------------------------------------
    # 96 (dense, p=1) + 336 (held_fixed, p=1) + 144 x 2/4 (routed) = 504
    assert report["read_bytes_per_token"] == 504
    assert report["read_gb_per_token"] == 504 / fp.GB
    assert report["breakdown"] == {
        "dense": 96,
        "routed": 72,
        "held_fixed": 336,
        "resident_codebooks": 0,
    }
    assert report["excluded"]["embedding_bytes"] == 256
    assert report["excluded"]["mtp_bytes"] == 128


def test_read_probability_is_topk_over_e_for_routed_and_one_for_dense(
    model_dir: Path, profile,
):
    report = _report(model_dir, profile)
    classes = report["classes"]
    assert classes["routed_experts"]["read_probability"] == TOPK / N_EXPERTS
    assert classes["dense"]["read_probability"] == 1.0
    assert classes["held_fixed"]["read_probability"] == 1.0
    assert classes["excluded_embedding"]["read_probability"] == 0.0
    assert report["routing"]["num_experts_per_tok"] == TOPK
    assert report["routing"]["n_routed_experts"] == N_EXPERTS
    assert report["routing"]["read_probability"] == TOPK / N_EXPERTS
    # The table is the single authority, and dense/held_fixed read every token.
    assert rt.READ_CLASS_TABLE["dense"] == 1.0
    assert rt.READ_CLASS_TABLE["held_fixed"] == 1.0


@pytest.mark.parametrize("topk", [1, 2, 3, 4])
def test_routed_read_bytes_scale_linearly_with_topk(
    model_dir: Path, profile, topk: int,
):
    config = json.loads((model_dir / "config.json").read_text())
    config["num_experts_per_tok"] = topk
    report = _report(model_dir, profile, config=config)
    assert report["breakdown"]["routed"] == round(
        FP8_EXPERT_STACK_BYTES * topk / N_EXPERTS)
    assert report["breakdown"]["dense"] == FP8_Q_PROJ_BYTES  # unchanged


def test_missing_topk_declaration_refuses(model_dir: Path, profile):
    config = json.loads((model_dir / "config.json").read_text())
    config.pop("num_experts_per_tok")
    with pytest.raises(rt.ReadTrafficError, match="declares no"):
        _report(model_dir, profile, config=config)


def test_expert_count_disagreement_refuses(model_dir: Path, profile):
    config = json.loads((model_dir / "config.json").read_text())
    config["num_experts"] = N_EXPERTS + 1
    with pytest.raises(rt.ReadTrafficError, match="carries 4 experts"):
        _report(model_dir, profile, config=config)


def test_classification_table(profile):
    """The mapping from tensor class to read probability, pinned."""
    cases = {
        "model.layers.0.mlp.experts.gate_up_proj": "routed_experts",
        "model.layers.0.mlp.experts.3.gate_proj": "routed_experts",
        "model.layers.0.mlp.shared_experts.gate_proj": "held_fixed",
        "model.layers.0.mlp.gate": "held_fixed",
        "model.layers.0.input_layernorm": "held_fixed",
        "lm_head": "held_fixed",
        "mtp.fc": "excluded_mtp",
        "mtp.layers.0.self_attn.q_proj": "excluded_mtp",
        "cb_codebook.ref0.NVFP4_CB_K12": "resident_codebooks",
    }
    for name, expected in cases.items():
        assert rt.classify_read_class(name, profile=profile) == expected, name
    # The embedding is the one class a single name cannot decide: untied it is
    # gathered (p=0), tied it IS the output projection (p=1).
    assert rt.classify_read_class(
        "model.embed_tokens", profile=profile,
        embedding_streamed=False) == "excluded_embedding"
    assert rt.classify_read_class(
        "model.embed_tokens", profile=profile,
        embedding_streamed=True) == "held_fixed"
    with pytest.raises(rt.ReadTrafficError, match="did not resolve"):
        rt.classify_read_class("model.embed_tokens", profile=profile)
    # A shared expert is read on EVERY token; the segment test is what keeps
    # it out of the routed class.
    assert rt.classify_read_class(
        "model.layers.0.mlp.experts.3.gate_proj", profile=profile,
        in_assignment=True) == "routed_experts"
    assert rt.classify_read_class(
        "model.layers.0.self_attn.q_proj", profile=profile,
        in_assignment=True) == "dense"


def test_exported_checkpoint_ledger(model_dir: Path, profile):
    """The post-export form classifies every shipped byte, or refuses."""
    report = rt.exported_checkpoint_read_traffic(
        str(model_dir), profile=profile)
    assert report["reconciliation"]["ledger_stored_bytes"] == SOURCE_TOTAL_BYTES
    classes = report["classes"]
    # q_proj 128 + router 64 + layernorm 16 + lm_head 256 = 464 always-active
    assert classes["held_fixed"]["stored_bytes"] == 464
    assert classes["routed_experts"]["stored_bytes"] == 256
    assert classes["excluded_embedding"]["stored_bytes"] == 256
    assert classes["excluded_mtp"]["stored_bytes"] == 128
    assert report["read_bytes_per_token"] == 464 + 256 * TOPK // N_EXPERTS


def test_claim_shape_is_advisory(model_dir: Path, profile):
    claim = rt.read_traffic_claim(str(model_dir), profile=profile)
    assert claim["value"] == pytest.approx(592 / fp.GB)
    assert claim["scope"] == rt.READ_SCOPE
    assert set(claim["breakdown"]) == {
        "dense", "routed", "held_fixed", "resident_codebooks"}
    # A broken input reports a reason; it never raises into an export.
    assert rt.read_traffic_claim(None)["value"] is None
    assert rt.read_traffic_claim("/nonexistent-export")["value"] is None


def test_exporter_shipcard_stamps_read_gb_per_token(model_dir: Path):
    """The stat lands beside `achieved_bpp` on the card the exporter writes."""
    from prismaquant import shipcard as _shipcard
    from prismaquant.export_native_compressed import _write_shipcard

    _write_shipcard(
        model_dir,
        source_model=str(model_dir),
        layer_config_path=None,
        assignment=ASSIGNMENT,
        config_assignment=ASSIGNMENT,
        hist={},
    )
    card = json.loads(
        (model_dir / _shipcard.SHIPCARD_FILENAME).read_text())
    build = card["build"]
    assert "achieved_bpp" in build
    claim = build["read_gb_per_token"]
    assert claim["value"] == pytest.approx(592 / fp.GB)
    assert claim["breakdown"]["routed"] == 128
    assert claim["routing"]["read_probability"] == TOPK / N_EXPERTS
    assert claim["scope"] == rt.READ_SCOPE


# ---------------------------------------------------------------------------
# Tied embeddings: the table IS the output projection and IS streamed.
# ---------------------------------------------------------------------------

def _tied_model_dir(tmp_path: Path, *, tie_flag: bool | None) -> Path:
    """The same model with no ``lm_head.weight`` -- i.e. tied.

    Source total drops by the 256 lm_head bytes to 848.
    """
    root = tmp_path / f"tied-{tie_flag}"
    root.mkdir()
    tensors = {k: v for k, v in _TENSORS.items() if k != "lm_head.weight"}
    _write_safetensors(root / "model.safetensors", tensors)
    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "num_experts": N_EXPERTS,
        "num_experts_per_tok": TOPK,
    }
    if tie_flag is not None:
        config["tie_word_embeddings"] = tie_flag
    (root / "config.json").write_text(json.dumps(config))
    return root


@pytest.mark.parametrize("tie_flag", [True, None])
def test_tied_embedding_is_streamed_every_token(tmp_path: Path, tie_flag):
    """A tied table is the logits projection: p=1, not excluded.

    Excluding it would drop one of the largest always-active tensors in the
    model (Qwen3-0.6B and LFM2.5 both tie), so this is a wrong published
    number rather than a conservative one.  ``tie_flag=None`` pins that the
    decision is made by OBSERVING that no output projection exists, not by
    reading a flag the config may omit.
    """
    root = _tied_model_dir(tmp_path, tie_flag=tie_flag)
    profile = detect_profile_with_warning(str(root), entrypoint="test")
    report = rt.assignment_read_traffic(
        ASSIGNMENT, STATS, model_path=str(root), profile=profile)

    assert report["embedding"]["streamed_per_token"] is True
    assert report["embedding"]["read_class"] == "held_fixed"
    assert report["embedding"]["lm_head_tensor_present"] is False
    assert report["embedding"]["config_tie_word_embeddings"] is tie_flag
    assert report["excluded"]["embedding_bytes"] == 0
    # router 64 + layernorm 16 + the tied embedding/lm_head 256 = 336
    assert report["classes"]["held_fixed"]["stored_bytes"] == 336
    assert report["classes"]["excluded_embedding"]["stored_bytes"] == 0
    # 848 source - 384 re-encoded source + 96 q_proj + 144 experts = 704
    assert report["reconciliation"]["ledger_stored_bytes"] == 704
    assert report["read_bytes_per_token"] == 96 + 72 + 336

    # Post-export there is no allocator/floor split, so every always-active
    # tensor is held_fixed: q_proj 128 + router 64 + ln 16 + embed 256 = 464.
    exported = rt.exported_checkpoint_read_traffic(str(root), profile=profile)
    assert exported["embedding"]["streamed_per_token"] is True
    assert exported["classes"]["held_fixed"]["stored_bytes"] == 464
    assert exported["read_bytes_per_token"] == 464 + 256 * TOPK // N_EXPERTS


def test_untied_embedding_is_excluded(model_dir: Path, profile):
    """The twin: with a real lm_head, the table is gathered, not streamed."""
    report = _report(model_dir, profile)
    assert report["embedding"]["streamed_per_token"] is False
    assert report["embedding"]["lm_head_tensor_present"] is True
    assert report["excluded"]["embedding_bytes"] == 256


def test_cb_codebook_sidecar_is_resident_not_traffic(model_dir: Path, profile):
    """A CB artifact ships its codebooks OUTSIDE the shard set.

    The sidecar is invisible to the safetensors ledger, so reporting zero
    resident codebook bytes for an artifact that ships them would be the same
    silent zero this module refuses everywhere else.  It is read from the
    artifact's own ``codebook_file`` declaration, counted as resident, and
    kept out of per-token traffic.
    """
    _write_safetensors(
        model_dir / "cb_codebooks.pqcb", {"cb_codebook.ref0.NVFP4_CB_K12": (4, 8)})
    (model_dir / "quant_config.json").write_text(
        json.dumps({"codebook_file": "cb_codebooks.pqcb"}))

    report = rt.exported_checkpoint_read_traffic(
        str(model_dir), profile=profile)
    assert report["breakdown"]["resident_codebooks"] == 64
    # Resident bytes are never per-token traffic, and never enter the shard
    # reconciliation either.
    assert report["read_bytes_per_token"] == 464 + 256 * TOPK // N_EXPERTS
    assert report["reconciliation"]["ledger_stored_bytes"] == SOURCE_TOTAL_BYTES
    assert report["reconciliation"][
        "codebook_sidecar_bytes_outside_shards"] == 64

    # A declaration with nothing behind it is a refusal, not a zero.
    (model_dir / "cb_codebooks.pqcb").unlink()
    with pytest.raises(rt.ReadTrafficError, match="codebook_file"):
        rt.exported_checkpoint_read_traffic(str(model_dir), profile=profile)


# ---------------------------------------------------------------------------
# Indexed lookups: three facts, because each weaker rule was falsified on a
# real artifact (see the module docstring).
# ---------------------------------------------------------------------------

def test_indexed_lookup_needs_all_three_facts(tmp_path: Path):
    """Integer dtype alone is a weight payload on both quantized lanes.

    The foils are the two real misclassifications this rule was hardened
    against: a scale-bearing packed weight, and a CB payload whose scales live
    in the codebook sidecar so only the artifact's declared ``targets`` names
    it.  Both are vocabulary-keyed integers, and both are read in full.
    """
    root = tmp_path / "lookups"
    root.mkdir()
    tensors = dict(_TENSORS)
    dtypes: dict[str, str] = {}

    # (1) the real lookup: integer, vocab-keyed, no scale, no target.
    tensors["model.layers.0.mlp.gate.tid2eid"] = (VOCAB, 2)      # 16*2*8 = 256
    dtypes["model.layers.0.mlp.gate.tid2eid"] = "I64"
    # (2) foil: a packed weight with a float scale sidecar.
    tensors["model.scaled_proj.weight_packed"] = (VOCAB, 4)      # 16*4*1 =  64
    dtypes["model.scaled_proj.weight_packed"] = "U8"
    tensors["model.scaled_proj.weight_scale"] = (VOCAB,)         # 16*4    =  64
    dtypes["model.scaled_proj.weight_scale"] = "F32"
    # (3) foil: a CB payload -- no scale in the shard set, declared instead.
    tensors["model.declared_proj.cb_qweight"] = (VOCAB, 4)       #          64
    dtypes["model.declared_proj.cb_qweight"] = "U8"
    # (4) an integer buffer keyed by something else: over-counted on purpose.
    tensors["model.layers.0.tiny_index"] = (4, 2)                #           8
    dtypes["model.layers.0.tiny_index"] = "I8"

    _write_safetensors(root / "model.safetensors", tensors, dtypes)
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "num_experts": N_EXPERTS,
        "num_experts_per_tok": TOPK,
        "vocab_size": VOCAB,
    }))
    (root / "quant_config.json").write_text(json.dumps({
        "config_groups": {
            "group_0": {"targets": ["model.declared_proj"]},
        },
    }))
    profile = detect_profile_with_warning(str(root), entrypoint="test")
    report = rt.exported_checkpoint_read_traffic(str(root), profile=profile)

    lookups = report["indexed_lookups"]
    assert lookups["vocab_size"] == VOCAB
    assert report["classes"]["excluded_indexed_lookup"]["n_tensors"] == 1
    assert report["excluded"]["indexed_lookup_bytes"] == 256
    # The three foils stay operands: 64 + 64 (its F32 scale is not integer)
    # + 64 + 8 -> only the integer ones are tallied.
    assert lookups["integer_bytes_read_in_full"] == 64 + 64 + 8
    # ... and they are streamed, not excluded.
    assert report["classes"]["held_fixed"]["stored_bytes"] == 464 + 64 + 64 + 64 + 8

    # The rule itself, stated directly.
    stems = rt.scaled_module_stems(rt._header_meta(str(root)))
    targets = rt.quantization_targets(str(root))
    def lookup(name, dtype, shape):
        return rt.is_indexed_lookup(
            dtype, shape, VOCAB, name=name, scaled_stems=stems,
            quant_targets=targets)
    assert lookup("model.layers.0.mlp.gate.tid2eid", "I64", (VOCAB, 2))
    assert not lookup("model.scaled_proj.weight_packed", "U8", (VOCAB, 4))
    assert not lookup("model.declared_proj.cb_qweight", "U8", (VOCAB, 4))
    assert not lookup("model.layers.0.tiny_index", "I8", (4, 2))
    # A float tensor is an operand whatever its shape (a BF16 lm_head is
    # vocabulary-keyed and streamed in full).
    assert not lookup("lm_head.weight", "BF16", (VOCAB, HIDDEN))
    # Facts that cannot be established never exclude: no vocab declaration,
    # or no way to check fact 3, means p=1 -- an over-count, never the reverse.
    assert not rt.is_indexed_lookup(
        "I64", (VOCAB, 2), None, name="x.tid2eid", scaled_stems=stems)
    assert not rt.is_indexed_lookup("I64", (VOCAB, 2), VOCAB, name="x.tid2eid")


def test_untied_declaration_without_an_lm_head_refuses(tmp_path: Path):
    """Nothing would carry the logits traffic; that is a contradiction."""
    root = _tied_model_dir(tmp_path, tie_flag=False)
    profile = detect_profile_with_warning(str(root), entrypoint="test")
    with pytest.raises(rt.ReadTrafficError, match="tie_word_embeddings=false"):
        rt.assignment_read_traffic(
            ASSIGNMENT, STATS, model_path=str(root), profile=profile)


# ---------------------------------------------------------------------------
# The checkpoint->live bridge is the shared name-projection layer.
# ---------------------------------------------------------------------------

def test_the_private_leaf_helper_is_gone():
    """The `.weight` leaf rule lives in the name-projection layer only.

    A second copy here is how namespace mappings drift apart between
    consumers; the mechanical pin keeps it deleted (R5: one mechanism).
    """
    assert not hasattr(rt, "_strip_weight")


def test_declared_out_of_graph_keys_route_through_the_layer(tmp_path: Path):
    """A key the profile DECLINES to map lands `excluded_non_text_graph`.

    The layer surfaces the profile's declared drop as
    ``ProjectedName.outcome == declared_out_of_graph`` -- data the classifier
    branches on -- so vision/audio bytes are itemized at p=0 rather than
    silently skipped or priced as always-active.
    """
    root = tmp_path / "with-visual"
    root.mkdir()
    tensors = dict(_TENSORS)
    tensors["model.visual.blocks.0.attn.q.weight"] = (HIDDEN, HIDDEN)  # 128 B
    _write_safetensors(root / "model.safetensors", tensors)
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "num_experts": N_EXPERTS,
        "num_experts_per_tok": TOPK,
    }))
    profile = detect_profile_with_warning(str(root), entrypoint="test")

    projected = npx.NameProjection(profile).checkpoint_to_live(
        "model.visual.blocks.0.attn.q.weight")
    assert projected.outcome == npx.DECLARED_OUT_OF_GRAPH

    # The classifier consumes exactly that outcome -- with and without a
    # caller-supplied projection instance.
    for kwargs in ({}, {"projection": npx.NameProjection(profile)}):
        assert rt.classify_read_class(
            "model.visual.blocks.0.attn.q", profile=profile,
            checkpoint_key="model.visual.blocks.0.attn.q.weight",
            **kwargs) == "excluded_non_text_graph"

    report = rt.exported_checkpoint_read_traffic(str(root), profile=profile)
    assert report["classes"]["excluded_non_text_graph"]["stored_bytes"] == 128
    # The ledger still partitions every shipped byte, visual keys included.
    assert report["reconciliation"]["ledger_stored_bytes"] == (
        SOURCE_TOTAL_BYTES + 128)


def test_a_failing_profile_mapping_refuses_rather_than_passing_through(
        model_dir: Path, profile):
    """A broken mapping accessor is a refusal, not a raw-key passthrough.

    The pre-projection code caught every accessor exception and fell back to
    the checkpoint key itself, so a misbehaving profile would have re-priced
    every unmappable tensor as an always-active operand instead of failing.
    Through the shared layer the refusal propagates; the advisory claim form
    reports it as a reason, never as a value.
    """
    class _BrokenMapping:
        def __getattr__(self, item):
            return getattr(profile, item)

        def checkpoint_to_live_name(self, ckpt_key, *, multimodal=False):
            raise RuntimeError("broken mapping")

    broken = _BrokenMapping()
    with pytest.raises(npx.NameProjectionError) as err:
        rt.exported_checkpoint_read_traffic(str(model_dir), profile=broken)
    assert err.value.code == "profile_accessor_failed"

    claim = rt.read_traffic_claim(str(model_dir), profile=broken)
    assert claim["value"] is None
    assert claim["source"] is None
    assert "profile_accessor_failed" in claim["reason"]
