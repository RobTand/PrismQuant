"""The CB export lane publishes the HF-standard shard layout — CPU-only.

Motivation: the shipped 87 GB DSv4 CB artifact is one ``model.safetensors``,
and a GB10 user stalled the default HF loader on a 128 GB unified-memory Spark
loading it, then resharded it by hand (RobTand/gridbook#47). The repo standard
is 1 GiB shards; this file pins the CB lane to it.

Serving a sharded CB artifact is not evidenced here: the CB lane serves only in
the pinned Gridbook runtime, which this suite may not import (`AGENTS.md:38`).
The evidence that the layout serves is the gridbook#47 reporter's own reshard,
plus the fact that this is the layout a stock HF/vLLM loader already reads.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

os.environ["PRISMAQUANT_CB_ENCODE_COMPILE"] = "0"

from prismaquant.export_nvfp4_cb import (  # noqa: E402
    export_nvfp4_cb as _export_nvfp4_cb,
)
from prismaquant.export_nvfp4_cb_streaming import (  # noqa: E402
    _StreamWriter,
    export_nvfp4_cb_streaming as _export_nvfp4_cb_streaming,
)
from prismaquant.shard_layout import (  # noqa: E402
    SHARD_INDEX_NAME,
    SINGLE_CONTAINER_NAME,
    TENSOR_PAYLOAD_IDENTITY_SCHEMA,
    container_names,
)
from prismaquant.shipcard import (  # noqa: E402
    compute_model_sha,
    load_shipcard,
    verify,
    CB_REQUIRED_SLOTS,
    REQUIRED_SLOTS,
)

# This module builds synthetic CB bodies on CPU and never serves them.
# Gridbook 0.9.1's v12 table names no CB cell on sm_121, so the route gate
# refuses these exports unless the artifact declares what it is.  See
# tests/cb_synthetic_target.py; the real sm_121 refusal stays asserted in
# tests/test_cb_route_status_gate.py.
pytestmark = pytest.mark.usefixtures("synthetic_cb_target")



def export_nvfp4_cb(*args, **kwargs):
    kwargs.setdefault("allow_unstamped_research", True)
    return _export_nvfp4_cb(*args, **kwargs)


def export_nvfp4_cb_streaming(*args, **kwargs):
    kwargs.setdefault("allow_unstamped_research", True)
    return _export_nvfp4_cb_streaming(*args, **kwargs)


# --- the streaming writer ---------------------------------------------------

def _writer(values: dict[str, torch.Tensor]) -> _StreamWriter:
    writer = _StreamWriter()
    for name, value in values.items():
        writer.add(
            name, value.dtype, tuple(value.shape),
            (lambda v=value: v.clone()),
        )
    return writer


def _payload(count: int, elements: int) -> dict[str, torch.Tensor]:
    return {
        f"t{i:02d}.weight": torch.full((elements,), i, dtype=torch.uint8)
        for i in range(count)
    }


def test_a_budget_above_the_artifact_keeps_the_legacy_single_container(tmp_path):
    values = _payload(4, 64)
    _writer(values).write(
        tmp_path / SINGLE_CONTAINER_NAME, shard_bytes=10 ** 9)

    assert (tmp_path / SINGLE_CONTAINER_NAME).is_file()
    assert not (tmp_path / SHARD_INDEX_NAME).exists()
    assert load_file(str(tmp_path / SINGLE_CONTAINER_NAME)).keys() == values.keys()


def test_a_small_budget_publishes_shards_and_an_index(tmp_path):
    values = _payload(6, 1000)
    writer = _writer(values)
    writer.write(tmp_path / SINGLE_CONTAINER_NAME, shard_bytes=2500)

    names = container_names(3)
    assert sorted(p.name for p in tmp_path.glob("*.safetensors")) == names
    assert not (tmp_path / SINGLE_CONTAINER_NAME).exists()

    index = json.loads((tmp_path / SHARD_INDEX_NAME).read_text())
    assert index["metadata"]["total_size"] == 6 * 1000
    assert index["weight_map"] == {
        "t00.weight": names[0], "t01.weight": names[0],
        "t02.weight": names[1], "t03.weight": names[1],
        "t04.weight": names[2], "t05.weight": names[2],
    }
    # Every tensor is readable, exactly once, at its indexed container.
    recovered: dict[str, torch.Tensor] = {}
    for name in names:
        shard = load_file(str(tmp_path / name))
        assert not set(shard) & set(recovered)
        recovered.update(shard)
    assert set(recovered) == set(values)
    for key, value in values.items():
        assert torch.equal(recovered[key], value)


def test_the_manifest_covers_every_published_container(tmp_path):
    writer = _writer(_payload(6, 1000))
    writer.write(tmp_path / SINGLE_CONTAINER_NAME, shard_bytes=2500)

    assert sorted(writer.last_weight_manifest_files) == container_names(3)
    for name, row in writer.last_weight_manifest_files.items():
        path = tmp_path / name
        assert row["bytes"] == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # The scalar pair describes ONE container and must not claim a shard set.
    assert writer.last_content_sha256 is None
    assert writer.last_content_bytes is None


def test_a_tensor_larger_than_the_budget_gets_its_own_container(tmp_path):
    values = {
        "small.weight": torch.zeros(8, dtype=torch.uint8),
        "big.weight": torch.zeros(4096, dtype=torch.uint8),
        "tail.weight": torch.zeros(8, dtype=torch.uint8),
    }
    _writer(values).write(tmp_path / SINGLE_CONTAINER_NAME, shard_bytes=1024)

    names = container_names(3)
    assert sorted(p.name for p in tmp_path.glob("*.safetensors")) == names
    assert list(load_file(str(tmp_path / names[1]))) == ["big.weight"]


def test_the_shard_layout_is_deterministic(tmp_path):
    values = _payload(9, 777)
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    a = _writer(values)
    b = _writer(values)
    a.write(left / SINGLE_CONTAINER_NAME, shard_bytes=2000)
    b.write(right / SINGLE_CONTAINER_NAME, shard_bytes=2000)

    assert a.last_weight_manifest_files == b.last_weight_manifest_files
    for name in container_names(len(a.last_weight_manifest_files)):
        assert (left / name).read_bytes() == (right / name).read_bytes()
    assert (left / SHARD_INDEX_NAME).read_text() == (
        right / SHARD_INDEX_NAME).read_text()


def test_the_payload_identity_survives_a_change_of_shard_budget(tmp_path):
    """The property a reshard must preserve: same tensors, same payload sha.

    ``compute_model_sha`` binds container filenames and sizes
    (``shipcard.compute_model_sha``), so it necessarily moves with the layout.
    The per-tensor digest does not, and it is what makes a resharded export
    recognisable as the same model.
    """
    values = _payload(6, 1000)
    one, many = tmp_path / "one", tmp_path / "many"
    one.mkdir()
    many.mkdir()
    single = _writer(values)
    sharded = _writer(values)
    single.write(one / SINGLE_CONTAINER_NAME, shard_bytes=10 ** 9)
    sharded.write(many / SINGLE_CONTAINER_NAME, shard_bytes=2500)

    assert single.last_tensor_content_sha256 == sharded.last_tensor_content_sha256
    assert set(single.last_tensor_content_sha256) == set(values)
    assert single.last_weight_manifest_files != sharded.last_weight_manifest_files


def test_the_pipeline_and_serial_paths_shard_identically(tmp_path, monkeypatch):
    values = _payload(6, 1000)
    serial, pipelined = tmp_path / "serial", tmp_path / "pipelined"
    serial.mkdir()
    pipelined.mkdir()

    monkeypatch.delenv("PRISMAQUANT_EXPORT_PIPELINE", raising=False)
    a = _writer(values)
    a.write(serial / SINGLE_CONTAINER_NAME, shard_bytes=2500)

    monkeypatch.setenv("PRISMAQUANT_EXPORT_PIPELINE", "1")
    b = _writer(values)
    b.write(pipelined / SINGLE_CONTAINER_NAME, shard_bytes=2500)

    assert a.last_weight_manifest_files == b.last_weight_manifest_files
    assert a.last_tensor_content_sha256 == b.last_tensor_content_sha256


def test_a_sharded_publication_refuses_a_stale_shard_run(tmp_path):
    """A leftover run at a different COUNT is indistinguishable to a globber."""
    (tmp_path / "model-00001-of-00007.safetensors").write_bytes(b"stale")
    writer = _writer(_payload(6, 1000))
    with pytest.raises(RuntimeError, match="unbound streaming resume"):
        writer.write(tmp_path / SINGLE_CONTAINER_NAME, shard_bytes=2500)
    assert not list(tmp_path.glob(".model-*.tmp"))


def test_a_failed_shard_leaves_no_partial_artifact(tmp_path):
    values = _payload(6, 1000)
    writer = _StreamWriter()
    for index, (name, value) in enumerate(values.items()):
        def producer(v=value, i=index):
            if i == 4:
                raise RuntimeError("producer failed on the last shard")
            return v.clone()
        writer.add(name, value.dtype, tuple(value.shape), producer)

    with pytest.raises(RuntimeError, match="producer failed"):
        writer.write(tmp_path / SINGLE_CONTAINER_NAME, shard_bytes=2500)

    assert list(tmp_path.iterdir()) == []
    assert writer.last_weight_manifest_files == {}


def test_before_publish_runs_once_after_the_last_producer(tmp_path):
    """Coverage is complete only when every shard's producers have run."""
    seen: list[int] = []
    values = _payload(6, 1000)
    writer = _StreamWriter()
    for name, value in values.items():
        writer.add(
            name, value.dtype, tuple(value.shape),
            (lambda v=value: (seen.append(len(seen)), v.clone())[1]),
        )

    calls: list[int] = []
    writer.write(
        tmp_path / SINGLE_CONTAINER_NAME,
        shard_bytes=2500,
        before_publish=lambda: calls.append(len(seen)),
    )
    assert calls == [len(values)]


def test_sharding_under_another_basename_is_refused(tmp_path):
    writer = _writer(_payload(6, 1000))
    with pytest.raises(ValueError, match="sharded publication is named by"):
        writer.write(tmp_path / "other.safetensors", shard_bytes=2500)


# --- the lane, end to end ---------------------------------------------------

def _write_model(mdl: Path, tensors: dict, hid: int = 256) -> None:
    mdl.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(mdl / SINGLE_CONTAINER_NAME))
    (mdl / "config.json").write_text(json.dumps({"hidden_size": hid}))


def _cb_fixture(workdir: Path):
    torch.manual_seed(0)
    mdl = workdir / "model"
    _write_model(mdl, {
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(256, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.self_attn.k_proj.weight":
            (torch.randn(256, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.experts.gate_up_proj.weight":
            (torch.randn(4, 128, 256) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    })
    assignment = workdir / "a.json"
    assignment.write_text(json.dumps({
        "model.layers.0.self_attn.q_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
        "model.layers.0.self_attn.k_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
        "model.layers.0.mlp.experts.gate_up_proj": {
            "data_type": "fp8_cb", "cb_k": 40},
    }))
    col_weights = {
        "model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05,
        "model.layers.0.self_attn.k_proj": torch.rand(256) + 0.05,
        "model.layers.0.mlp.experts.gate_up_proj": torch.rand(4, 1, 256) + 0.05,
    }
    return mdl, assignment, col_weights


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_both_cb_exporters_publish_the_same_tensors_sharded_or_not(
    tmp_path, exporter,
):
    mdl, assignment, col_weights = _cb_fixture(tmp_path)
    one = tmp_path / "one"
    many = tmp_path / "many"
    exporter(mdl, assignment, one, col_weights, device="cpu",
             shard_bytes=10 ** 9)
    exporter(mdl, assignment, many, col_weights, device="cpu",
             shard_bytes=32 * 1024)

    assert (one / SINGLE_CONTAINER_NAME).is_file()
    assert not (one / SHARD_INDEX_NAME).exists()

    shards = sorted(p.name for p in many.glob("*.safetensors"))
    assert len(shards) > 1, "the small budget must actually shard this fixture"
    assert shards == container_names(len(shards))
    assert not (many / SINGLE_CONTAINER_NAME).exists()

    index = json.loads((many / SHARD_INDEX_NAME).read_text())
    assert sorted(set(index["weight_map"].values())) == shards

    flat = load_file(str(one / SINGLE_CONTAINER_NAME))
    recovered: dict[str, torch.Tensor] = {}
    for name in shards:
        recovered.update(load_file(str(many / name)))
    assert set(recovered) == set(flat)
    for key, value in flat.items():
        assert torch.equal(recovered[key], value), key
    assert index["metadata"]["total_size"] == sum(
        t.numel() * t.element_size() for t in flat.values()
    )
    assert set(index["weight_map"]) == set(flat)


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_a_sharded_cb_artifact_carries_a_valid_card_and_inventory(
    tmp_path, exporter,
):
    mdl, assignment, col_weights = _cb_fixture(tmp_path)
    out = tmp_path / "sharded"
    exporter(mdl, assignment, out, col_weights, device="cpu",
             shard_bytes=32 * 1024)

    card = load_shipcard(out / "shipcard.json")
    assert card["model_sha"] == compute_model_sha(out)
    assert verify(card, model_dir=out) == [
        f"{slot}: UNFILLED" for slot in REQUIRED_SLOTS + CB_REQUIRED_SLOTS
    ]
    assert card["artifact_bytes"] == sum(
        path.stat().st_size
        for pattern in ("*.safetensors", "*.pqcb")
        for path in out.glob(pattern)
    )

    config = json.loads((out / "quant_config.json").read_text())
    inventory = config["provenance"]["artifact_inventory"]
    files = {
        path.relative_to(out).as_posix(): path.stat().st_size
        for path in out.rglob("*") if path.is_file()
    }
    assert inventory["file_bytes"] == files
    # The index is artifact content and is measured like everything else.
    assert SHARD_INDEX_NAME in inventory["file_bytes"]
    assert inventory["export_directory_bytes"] == sum(files.values())

    manifest = config["provenance"]["weight_content_manifest"]
    on_disk = sorted(p.name for p in out.glob("*.safetensors"))
    assert sorted(manifest["files"]) == on_disk
    for name, row in manifest["files"].items():
        path = out / name
        assert row["bytes"] == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_both_exporters_stamp_a_budget_invariant_payload_identity(
    tmp_path, exporter,
):

    """The property `model_sha` cannot carry across a reshard."""
    mdl, assignment, col_weights = _cb_fixture(tmp_path)
    one, many = tmp_path / "one", tmp_path / "many"
    exporter(mdl, assignment, one, col_weights, device="cpu",
             shard_bytes=10 ** 9)
    exporter(mdl, assignment, many, col_weights, device="cpu",
             shard_bytes=32 * 1024)

    def identity(out: Path) -> dict:
        config = json.loads((out / "quant_config.json").read_text())
        return config["provenance"]["tensor_payload_identity"]

    single, sharded = identity(one), identity(many)
    assert single["schema"] == TENSOR_PAYLOAD_IDENTITY_SCHEMA
    assert single["tensors"] == len(
        load_file(str(one / SINGLE_CONTAINER_NAME)))
    assert single == sharded
    # ...while the file-scoped identity necessarily moved with the layout.
    assert compute_model_sha(one) != compute_model_sha(many)


def test_the_two_exporters_agree_on_the_payload_identity(tmp_path):
    """One identity for one tensor payload, whichever exporter wrote it."""
    mdl, assignment, col_weights = _cb_fixture(tmp_path)
    batch, streamed = tmp_path / "batch", tmp_path / "streamed"
    export_nvfp4_cb(mdl, assignment, batch, col_weights, device="cpu",
                    shard_bytes=32 * 1024)
    export_nvfp4_cb_streaming(mdl, assignment, streamed, col_weights,
                              device="cpu", shard_bytes=32 * 1024)

    def identity(out: Path) -> dict:
        config = json.loads((out / "quant_config.json").read_text())
        return config["provenance"]["tensor_payload_identity"]

    assert identity(batch) == identity(streamed)


def test_a_sharded_cb_artifact_passes_the_publish_freeze(tmp_path):
    """`publish_artifact` binds the frozen safetensors set to the manifest."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import publish_artifact  # noqa: E402

    mdl, assignment, col_weights = _cb_fixture(tmp_path)
    out = tmp_path / "sharded"
    export_nvfp4_cb_streaming(
        mdl, assignment, out, col_weights, device="cpu",
        shard_bytes=32 * 1024)
    assert len(list(out.glob("*.safetensors"))) > 1

    expected = hashlib.sha256(
        (out / "shipcard.json").read_bytes()).hexdigest()
    snapshot = publish_artifact._freeze_artifact(
        out, expected_shipcard_sha256=expected)
    try:
        publish_artifact._verify_declared_weight_hashes(snapshot)
        frozen = {
            entry.relative_path for entry in snapshot.entries
            if entry.relative_path.endswith(".safetensors")
        }
        assert frozen == {p.name for p in out.glob("*.safetensors")}
        assert SHARD_INDEX_NAME in {
            entry.relative_path for entry in snapshot.entries
        }
    finally:
        snapshot.close()


def test_the_inventory_refuses_a_stale_index_beside_a_single_container(tmp_path):
    mdl, assignment, col_weights = _cb_fixture(tmp_path)
    out = tmp_path / "one"
    export_nvfp4_cb_streaming(mdl, assignment, out, col_weights, device="cpu",
                              shard_bytes=10 ** 9)
    (out / SHARD_INDEX_NAME).write_text(json.dumps({
        "metadata": {"total_size": 1},
        "weight_map": {"a.weight": "model-00001-of-00002.safetensors"},
    }))

    from prismaquant.nvfp4_cb_footprint import cb_export_artifact_inventory

    config = json.loads((out / "quant_config.json").read_text())
    with pytest.raises(AssertionError, match="unexpected/stale"):
        cb_export_artifact_inventory(
            out,
            serialized_payload=config["provenance"]["serialized_payload"],
            cb_tensor_names=[],
            codebook_file=config.get("codebook_file"),
            expected_model_files=[SINGLE_CONTAINER_NAME],
        )


def test_the_inventory_refuses_a_shard_run_with_no_index(tmp_path):
    mdl, assignment, col_weights = _cb_fixture(tmp_path)
    out = tmp_path / "sharded"
    export_nvfp4_cb_streaming(mdl, assignment, out, col_weights, device="cpu",
                              shard_bytes=32 * 1024)
    containers = sorted(p.name for p in out.glob("*.safetensors"))
    (out / SHARD_INDEX_NAME).unlink()

    from prismaquant.nvfp4_cb_footprint import cb_export_artifact_inventory

    config = json.loads((out / "quant_config.json").read_text())
    with pytest.raises(AssertionError, match="missing model.safetensors.index"):
        cb_export_artifact_inventory(
            out,
            serialized_payload=config["provenance"]["serialized_payload"],
            cb_tensor_names=[],
            codebook_file=config.get("codebook_file"),
            expected_model_files=containers,
        )
