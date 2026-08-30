"""CPU-only tests for tools/tp2_budget_plan.py.

Every fixture is a synthetic safetensors file written into a worktree-local
temp dir: no artifact, no GPU, no network, and nothing written under /tmp.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import tempfile

import pytest

from tools import tp2_budget_plan as T


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def workdir():
    """A scratch dir INSIDE the worktree (project rule: never write to /tmp)."""
    root = REPO_ROOT / "tests" / "_tmp_tp2"
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            root.rmdir()
        except OSError:
            pass


def write_safetensors(path: Path, tensors, *, corrupt_offsets_for=None) -> None:
    """tensors: {name: (dtype, shape)}.  Writes a real (tiny) safetensors file."""
    header = {}
    offset = 0
    blobs = []
    for name, (dtype, shape) in tensors.items():
        nbytes = T.tensor_bytes(dtype, shape) if dtype in T.DTYPE_BYTES else 0
        span = nbytes
        if corrupt_offsets_for == name:
            span = nbytes + 1
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + span]}
        offset += span
        blobs.append(b"\0" * span)
    blob = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for b in blobs:
            fh.write(b)


def simple_model(num_layers: int, per_layer_elems, *, prefix="model.") -> dict:
    """A minimal well-formed checkpoint: embed + N layers + norm + lm_head."""
    tensors = {
        f"{prefix}embed_tokens.weight": ("BF16", [8, 4]),
        f"{prefix}norm.weight": ("F32", [4]),
        "lm_head.weight": ("BF16", [8, 4]),
    }
    for i in range(num_layers):
        elems = per_layer_elems[i] if isinstance(per_layer_elems, (list, tuple)) \
            else per_layer_elems
        tensors[f"{prefix}layers.{i}.mlp.weight"] = ("U8", [elems])
    return tensors


# ---------------------------------------------------------------------------
# 1. Exact byte math across dtypes.


@pytest.mark.parametrize("dtype,width", [
    ("F32", 4), ("BF16", 2), ("F16", 2), ("F8_E4M3", 1), ("F8_E5M2", 1),
    ("F8_E8M0", 1), ("U8", 1), ("I8", 1), ("I64", 8), ("I32", 4), ("BOOL", 1),
])
def test_tensor_bytes_exact_per_dtype(dtype, width):
    assert T.tensor_bytes(dtype, [7, 3]) == 21 * width
    assert T.tensor_bytes(dtype, []) == width  # 0-d tensor is one element


def test_unknown_dtype_is_a_hard_error():
    with pytest.raises(T.HeaderError, match="unknown safetensors dtype"):
        T.tensor_bytes("F4_E2M1", [16])


def test_artifact_byte_totals_are_exact_across_dtypes(workdir):
    tensors = {
        "model.embed_tokens.weight": ("BF16", [10, 4]),      # 80
        "model.layers.0.a.cb_qweight": ("U8", [100]),        # 100
        "model.layers.0.a.weight_scale": ("F32", [10]),      # 40
        "model.layers.0.a.act_scale": ("F8_E4M3", [10]),     # 10
        "model.layers.0.a.idx": ("I8", [10]),                # 10
        "model.norm.weight": ("F32", [4]),                   # 16
        "lm_head.weight": ("BF16", [10, 4]),                 # 80
    }
    write_safetensors(workdir / "model.safetensors", tensors)
    collected = T.collect_artifact_tensors(workdir)
    classified = T.classify_entries(collected["entries"])
    total = sum(e["bytes"] for e in classified)
    assert total == 80 + 100 + 40 + 10 + 10 + 16 + 80
    assert T.bucket_bytes(classified, T.BUCKET_EMBED) == 80
    assert T.bucket_bytes(classified, T.BUCKET_HEAD) == 80
    assert T.bucket_bytes(classified, T.BUCKET_NORM) == 16
    assert T.layer_byte_table(classified) == [160]


def test_offset_mismatch_is_a_hard_error(workdir):
    write_safetensors(workdir / "model.safetensors",
                      {"model.layers.0.a.weight": ("BF16", [4, 4])},
                      corrupt_offsets_for="model.layers.0.a.weight")
    with pytest.raises(T.HeaderError, match="data_offsets span"):
        T.collect_artifact_tensors(workdir)


def test_tensor_data_is_never_read(workdir, monkeypatch):
    """The header parse must not depend on the data region existing."""
    path = workdir / "model.safetensors"
    header = {"model.layers.0.a.weight": {"dtype": "BF16", "shape": [1024, 1024],
                                          "data_offsets": [0, 2 * 1024 * 1024]}}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:          # header only; no tensor bytes at all
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
    entries = T.collect_artifact_tensors(workdir)["entries"]
    assert entries[0]["bytes"] == 2 * 1024 * 1024
    assert path.stat().st_size < 4096     # proof no data region was needed


# ---------------------------------------------------------------------------
# 2. Classification rules.


def test_layer_regex_matches_dsv4_unprefixed_names():
    """DSv4 ships `layers.0.attn...`; a bare `\\.layers\\.` matches none of it."""
    assert T.classify("layers.0.attn.wo_b.cb_qweight")["layer"] == 0
    assert T.classify("model.language_model.layers.63.mlp.w")["layer"] == 63


def test_endcap_placement():
    assert T.classify("model.language_model.embed_tokens.weight")["bucket"] == T.BUCKET_EMBED
    assert T.classify("embed.weight")["bucket"] == T.BUCKET_EMBED
    assert T.classify("lm_head.weight")["bucket"] == T.BUCKET_HEAD
    assert T.classify("head.weight")["bucket"] == T.BUCKET_HEAD
    assert T.classify("model.norm.weight")["bucket"] == T.BUCKET_NORM
    assert T.classify("model.language_model.norm.weight")["bucket"] == T.BUCKET_NORM


def test_head_like_name_is_labelled_assumed_placement():
    cls = T.classify("hc_head_base")
    assert cls["bucket"] == T.BUCKET_HEAD
    assert cls["assumed_placement"] is True
    assert T.classify("head.weight")["assumed_placement"] is False


def test_sidecar_rule_fires_before_the_layer_and_norm_rules():
    """mtp.layers.N.* must NOT become a body layer; mtp.norm the final norm."""
    assert T.classify("mtp.layers.0.mlp.down_proj.weight")["bucket"] == T.BUCKET_SIDECAR
    assert T.classify("mtp.norm.weight")["bucket"] == T.BUCKET_SIDECAR
    assert T.classify("mtp.pre_fc_norm_embedding.weight")["bucket"] == T.BUCKET_SIDECAR
    assert T.classify("model.nextn.layers.0.w")["bucket"] == T.BUCKET_SIDECAR
    assert T.classify("draft_model.layers.0.w")["bucket"] == T.BUCKET_SIDECAR


def test_unmatched_names_are_a_hard_error_listing_all_of_them():
    entries = [
        {"name": "model.layers.0.a.weight", "bytes": 4},
        {"name": "mystery.tensor.one", "bytes": 4},
        {"name": "another.unknown.thing", "bytes": 4},
    ]
    with pytest.raises(T.ClassificationError) as exc:
        T.classify_entries(entries)
    msg = str(exc.value)
    assert "2 tensor name(s) matched no pipeline-stage rule" in msg
    assert "mystery.tensor.one" in msg
    assert "another.unknown.thing" in msg


def test_non_contiguous_layer_ids_are_a_hard_error():
    classified = [
        {"name": "layers.0.a", "bytes": 4, "bucket": T.BUCKET_LAYER, "layer": 0},
        {"name": "layers.2.a", "bytes": 4, "bucket": T.BUCKET_LAYER, "layer": 2},
    ]
    with pytest.raises(T.ClassificationError, match="not contiguous"):
        T.layer_byte_table(classified)


# ---------------------------------------------------------------------------
# 3. Splits.


def test_vllm_even_split_remainder_goes_to_partitions_minus_two():
    assert T.vllm_even_split(64, 2) == [32, 32]
    assert T.vllm_even_split(43, 2) == [22, 21]     # DSv4: extra on rank 0
    assert T.vllm_even_split(10, 4) == [2, 3, 3, 2]  # skips first and last
    assert sum(T.vllm_even_split(61, 4)) == 61


def test_even_split_refuses_more_stages_than_layers():
    with pytest.raises(ValueError, match="cannot fill"):
        T.vllm_even_split(3, 4)
    with pytest.raises(ValueError, match="cannot fill"):
        T.optimal_split([1, 1, 1], 4)


def test_optimal_split_beats_even_on_an_imbalanced_model():
    # Front-loaded body: the even 3/3 split is far worse than 4/2.
    layer_bytes = [10, 10, 10, 10, 40, 40]
    even = T.vllm_even_split(len(layer_bytes), 2)
    opt = T.optimal_split(layer_bytes, 2)
    assert even == [3, 3]
    assert opt == [4, 2]
    even_ranks = T.rank_weight_bytes(even, layer_bytes, 0, 0)
    opt_ranks = T.rank_weight_bytes(opt, layer_bytes, 0, 0)
    assert even_ranks == [30, 90]
    assert opt_ranks == [40, 80]
    assert max(opt_ranks) <= max(even_ranks)


def test_optimal_split_accounts_for_the_endcaps():
    layer_bytes = [10] * 4
    # A fat last-stage endcap pulls layers off the last rank.
    assert T.optimal_split(layer_bytes, 2, first_extra=0, last_extra=20) == [3, 1]
    assert T.optimal_split(layer_bytes, 2, first_extra=20, last_extra=0) == [1, 3]
    assert T.optimal_split(layer_bytes, 2, 0, 0) == [2, 2]


def test_optimal_split_is_deterministic_under_ties():
    layer_bytes = [10] * 6
    assert T.optimal_split(layer_bytes, 3) == [2, 2, 2]
    # Earliest-boundary tie-break: every rank must still hold >= 1 layer.
    assert min(T.optimal_split([1, 1, 1, 100], 3)) >= 1


def test_rank_weight_bytes_places_endcaps_on_first_and_last(workdir):
    tensors = simple_model(4, [100, 100, 100, 100])
    tensors["mtp.fc.weight"] = ("BF16", [50])   # 100 B sidecar -> last stage
    write_safetensors(workdir / "model.safetensors", tensors)
    plan = T.build_plan(workdir, "pp", 2, 121.0, None, "optimal")
    inv = plan["inventory"]
    assert inv["embed_bytes"] == 64 and inv["head_bytes"] == 64
    assert inv["final_norm_bytes"] == 16
    assert inv["sidecar_bytes"] == 100
    endcaps = plan["pp"]["endcaps"]
    assert endcaps["first_stage_bytes"] == 64            # embed only
    assert endcaps["last_stage_bytes"] == 64 + 16 + 100  # head + norm + sidecar
    rows = plan["residency_rows"]
    assert rows[0]["weight_bytes"] - sum(inv["layer_bytes"][:rows[0]["layer_count"]]) == 64
    assert plan["sidecar_line_items"][0]["name"] == "mtp.fc.weight"


# ---------------------------------------------------------------------------
# 4. Feasibility arithmetic (hand-computed fixture).


def test_max_feasible_matches_the_hand_computed_fixture():
    layer_bytes = [10, 10, 10, 10]
    counts = T.optimal_split(layer_bytes, 2, first_extra=5, last_extra=7)
    assert counts == [2, 2]
    mf = T.max_feasible_artifact_bytes(
        counts, layer_bytes, first_extra=5, last_extra=7,
        budget_bytes=100, overhead_bytes=20)
    # rank0: (100-20-5)/(40*0.5) = 3.75 ; rank1: (100-20-7)/20 = 3.65 -> binding
    assert mf["feasible"] is True
    assert mf["binding_rank"] == 1
    assert mf["s_max"] == pytest.approx(3.65)
    assert mf["max_body_bytes"] == pytest.approx(146.0)
    assert mf["fixed_endcap_bytes"] == 12
    assert mf["max_artifact_bytes"] == pytest.approx(158.0)


def test_max_feasible_is_infeasible_when_endcaps_alone_bust_the_budget():
    mf = T.max_feasible_artifact_bytes([1, 1], [10, 10], first_extra=0,
                                       last_extra=200, budget_bytes=100,
                                       overhead_bytes=20)
    assert mf["feasible"] is False
    assert mf["binding_rank"] == 1


def test_missing_overhead_prints_unknown_not_a_number(workdir):
    write_safetensors(workdir / "model.safetensors", simple_model(4, 100))
    plan = T.build_plan(workdir, "pp", 2, 121.0, None, "optimal")
    assert plan["max_feasible"]["label"] == T.UNKNOWN_OVERHEAD
    for row in plan["residency_rows"]:
        assert row["headroom_bytes"] is None
        assert row["headroom_label"] == T.UNKNOWN_OVERHEAD
    text = T.render(plan)
    assert text.count(T.UNKNOWN_OVERHEAD) >= 3
    assert "MAX-FEASIBLE-ARTIFACT: UNKNOWN-NEEDS-OVERHEAD" in text


def test_max_feasible_render_is_labelled_assumed_linear(workdir):
    write_safetensors(workdir / "model.safetensors", simple_model(4, 100))
    plan = T.build_plan(workdir, "pp", 2, 121.0, 1.0, "optimal")
    text = T.render(plan)
    assert T.LABEL_ASSUMED_LINEAR in text
    assert "body bytes scale linearly" in text


def test_pp_partition_flag_selects_which_split_drives_the_table(workdir):
    tensors = simple_model(6, [10, 10, 10, 10, 400, 400])
    write_safetensors(workdir / "model.safetensors", tensors)
    even = T.build_plan(workdir, "pp", 2, 121.0, 1.0, "even")
    opt = T.build_plan(workdir, "pp", 2, 121.0, 1.0, "optimal")
    # Both splits are always reported, whichever one is selected.
    assert set(even["pp"]["splits"]) == {"even", "optimal"}
    assert even["pp"]["selected"] == "even"
    assert opt["pp"]["selected"] == "optimal"
    assert even["pp"]["splits"]["even"]["counts"] == [3, 3]
    assert opt["pp"]["splits"]["optimal"]["counts"] != [3, 3]
    assert (opt["pp"]["splits"]["optimal"]["max_rank_bytes"]
            < even["pp"]["splits"]["even"]["max_rank_bytes"])
    assert opt["residency_rows"][0]["layer_count"] == \
        opt["pp"]["splits"]["optimal"]["counts"][0]
    assert "VLLM_PP_LAYER_PARTITION=" in T.render(opt)


# ---------------------------------------------------------------------------
# 5. TP mode: everything ASSUMED.


def test_tp_mode_labels_every_line_assumed(workdir):
    write_safetensors(workdir / "model.safetensors", simple_model(4, 100))
    plan = T.build_plan(workdir, "tp", 2, 121.0, 4.0, "optimal")
    assert plan["tp"]["label"] == T.LABEL_ASSUMED_TP
    for row in plan["residency_rows"]:
        assert row["label"] == T.LABEL_ASSUMED_TP
        assert row["headroom_label"] == T.LABEL_ASSUMED_TP
    text = T.render(plan)
    # Every printed residency row carries the label, not just the section.
    table = text.split("PER-RANK RESIDENCY")[1].split("TP DECOMPOSITION")[0]
    row_lines = [ln for ln in table.splitlines()
                 if ln.strip() and ln.split()[0].isdigit()]
    assert len(row_lines) == 2
    assert all("ASSUMED" in ln for ln in row_lines)
    assert "TP DECOMPOSITION" in text
    assert "ASSUMED sharded" in text and "ASSUMED replicated" in text
    assert "Every TP row above is ASSUMED" in text
    # A max-feasible number is refused, not guessed, in TP mode.
    assert plan["max_feasible"]["label"] == "NOT-COMPUTED-FOR-TP"


def test_tp_assumed_shard_classification():
    assert T.tp_assumed_replicated("model.layers.0.input_layernorm.weight",
                                   T.BUCKET_LAYER) is True
    assert T.tp_assumed_replicated("model.layers.0.mlp.up_proj.bias",
                                   T.BUCKET_LAYER) is True
    assert T.tp_assumed_replicated("model.layers.0.mlp.up_proj.weight",
                                   T.BUCKET_LAYER) is False
    assert T.tp_assumed_replicated("model.embed_tokens.weight",
                                   T.BUCKET_EMBED) is True
    assert T.tp_assumed_replicated("lm_head.weight", T.BUCKET_HEAD) is True


def test_tp_per_rank_arithmetic(workdir):
    tensors = {
        "model.embed_tokens.weight": ("U8", [64]),            # 64 replicated
        "model.layers.0.mlp.up_proj.weight": ("U8", [400]),   # 400 sharded
        "model.layers.0.input_layernorm.weight": ("U8", [8]),  # 8 replicated
        "model.norm.weight": ("U8", [8]),                      # 8 replicated
        "lm_head.weight": ("U8", [64]),                        # 64 replicated
    }
    write_safetensors(workdir / "model.safetensors", tensors)
    plan = T.build_plan(workdir, "tp", 2, 121.0, None, "optimal")
    tp = plan["tp"]
    assert tp["assumed_sharded_bytes"] == 400
    assert tp["assumed_replicated_bytes"] == 64 + 8 + 8 + 64
    assert tp["assumed_per_rank_bytes"] == pytest.approx(400 / 2 + 144)


# ---------------------------------------------------------------------------
# 6. Sharded-index path.


def test_sharded_index_path_sums_across_shards(workdir):
    a = {"model.embed_tokens.weight": ("BF16", [8, 4]),
         "model.layers.0.mlp.weight": ("U8", [100])}
    b = {"model.layers.1.mlp.weight": ("U8", [100]),
         "model.norm.weight": ("F32", [4]),
         "lm_head.weight": ("BF16", [8, 4])}
    write_safetensors(workdir / "model-00001-of-00002.safetensors", a)
    write_safetensors(workdir / "model-00002-of-00002.safetensors", b)
    weight_map = {k: "model-00001-of-00002.safetensors" for k in a}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in b})
    total = 64 + 100 + 100 + 16 + 64
    (workdir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}))
    plan = T.build_plan(workdir, "pp", 2, 121.0, None, "optimal")
    assert plan["inventory"]["total_classified_bytes"] == total
    assert plan["inventory"]["num_layers"] == 2
    assert len(plan["inventory"]["index"]["shards"]) == 2


def test_index_total_size_mismatch_is_a_hard_error(workdir):
    tensors = {"model.embed_tokens.weight": ("BF16", [8, 4]),
               "model.layers.0.mlp.weight": ("U8", [100]),
               "model.norm.weight": ("F32", [4]),
               "lm_head.weight": ("BF16", [8, 4])}
    write_safetensors(workdir / "model-00001-of-00001.safetensors", tensors)
    (workdir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 999999},
         "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}))
    with pytest.raises(T.HeaderError, match="metadata.total_size"):
        T.collect_artifact_tensors(workdir)


def test_index_and_shard_header_disagreement_is_a_hard_error(workdir):
    tensors = {"model.layers.0.mlp.weight": ("U8", [100])}
    write_safetensors(workdir / "model-00001-of-00001.safetensors", tensors)
    weight_map = {"model.layers.0.mlp.weight": "model-00001-of-00001.safetensors",
                  "model.layers.9.ghost.weight": "model-00001-of-00001.safetensors"}
    (workdir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    with pytest.raises(T.HeaderError, match="weight_map and shard headers disagree"):
        T.collect_artifact_tensors(workdir)


def test_missing_checkpoint_is_a_hard_error(workdir):
    with pytest.raises(T.HeaderError, match="neither model.safetensors.index.json"):
        T.collect_artifact_tensors(workdir)


# ---------------------------------------------------------------------------
# 7. Non-safetensors payload reporting + CLI plumbing.


def test_non_safetensors_files_are_reported_not_classified(workdir):
    write_safetensors(workdir / "model.safetensors", simple_model(2, 100))
    (workdir / "cb_codebooks.pqcb").write_bytes(b"\0" * 4096)
    (workdir / "config.json").write_text("{}")
    plan = T.build_plan(workdir, "pp", 2, 121.0, None, "optimal")
    names = {r["file"]: r for r in plan["reported_not_classified"]}
    assert names["cb_codebooks.pqcb"]["bytes"] == 4096
    assert names["cb_codebooks.pqcb"]["kind"] == "payload"
    assert names["config.json"]["kind"] == "metadata-extension"
    # Excluded from residency: totals must not move.
    assert plan["inventory"]["total_classified_bytes"] == 64 + 200 + 16 + 64
    text = T.render(plan)
    assert "REPORTED-NOT-CLASSIFIED" in text
    assert "cb_codebooks.pqcb" in text


def test_parse_parallelism():
    assert T.parse_parallelism("pp:2") == ("pp", 2)
    assert T.parse_parallelism("TP:4") == ("tp", 4)
    for bad in ("pp", "dp:2", "pp:1", "pp:x"):
        with pytest.raises(SystemExit):
            T.parse_parallelism(bad)


def test_json_out_refuses_tmp():
    with pytest.raises(SystemExit, match="refusing to write under /tmp"):
        T.check_out_path(Path("/tmp/plan.json"))
    with pytest.raises(SystemExit):
        T.check_out_path(Path("/tmp/nested/dir/plan.json"))


def test_cli_end_to_end_writes_json(workdir, capsys):
    write_safetensors(workdir / "model.safetensors", simple_model(4, 100))
    out = workdir / "plan.json"
    rc = T.main(["--artifact", str(workdir), "--parallelism", "pp:2",
                 "--overhead-gb-per-rank", "2", "--json-out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["schema"] == T.SCHEMA
    assert payload["parallelism"] == {"mode": "pp", "world_size": 2}
    assert payload["max_feasible"]["label"] == T.LABEL_ASSUMED_LINEAR
    text = capsys.readouterr().out
    assert "PER-RANK RESIDENCY" in text and "GiB = 1024**3" in text


def test_cli_rejects_an_unclassifiable_artifact(workdir, capsys):
    tensors = simple_model(2, 100)
    tensors["some.unknown.tensor"] = ("BF16", [4])
    write_safetensors(workdir / "model.safetensors", tensors)
    rc = T.main(["--artifact", str(workdir), "--parallelism", "pp:2"])
    assert rc == 2
    assert "some.unknown.tensor" in capsys.readouterr().err
