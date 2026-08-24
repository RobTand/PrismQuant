"""Three-stage CB stream execution tests.

CPU cases use the streaming writer's fake-producer seam: serialization is real
while reads/encodes are tiny deterministic callables. The GPU case exercises
the complete streaming CB exporter and is intentionally deferred on CPU hosts.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

os.environ["PRISMAQUANT_CB_ENCODE_COMPILE"] = "0"

from prismaquant.export_nvfp4_cb_streaming import (  # noqa: E402
    _StreamWriter,
    export_nvfp4_cb_streaming,
)
from prismaquant.shipcard import load_shipcard  # noqa: E402


def _fake_writer(
    values: list[torch.Tensor],
    *,
    delays: list[float] | None = None,
    completions: list[int] | None = None,
    fail_index: int | None = None,
) -> _StreamWriter:
    writer = _StreamWriter()
    delays = delays or [0.0] * len(values)
    completion_lock = threading.Lock()
    for index, value in enumerate(values):
        canonical = value.clone()

        def producer(tensor=canonical):
            return tensor.clone()

        def reader(tensor=canonical):
            return tensor.clone()

        def encoder(source, i=index, delay=delays[index]):
            time.sleep(delay)
            if fail_index == i:
                raise RuntimeError(f"fake encoder failed for tensor {i}")
            if completions is not None:
                with completion_lock:
                    completions.append(i)
            return source.contiguous()

        writer.add(
            f"tensor_{index:02d}",
            canonical.dtype,
            canonical.shape,
            producer,
            reader=reader,
            encoder=encoder,
        )
    return writer


def test_pipeline_bytes_match_serial_with_out_of_order_encode(
    tmp_path: Path,
    monkeypatch,
):
    values = [
        torch.arange(17, dtype=torch.float32),
        torch.arange(9, dtype=torch.int64),
        torch.arange(31, dtype=torch.uint8),
        torch.linspace(-1, 1, 13, dtype=torch.float32),
    ]
    serial = tmp_path / "serial.safetensors"
    pipelined = tmp_path / "pipelined.safetensors"

    monkeypatch.delenv("PRISMAQUANT_EXPORT_PIPELINE", raising=False)
    _fake_writer(values).write(serial)

    completions: list[int] = []
    monkeypatch.setenv("PRISMAQUANT_EXPORT_PIPELINE", "1")
    monkeypatch.setenv("PRISMAQUANT_EXPORT_PREFETCH_DEPTH", "3")
    _fake_writer(
        values,
        delays=[0.08, 0.0, 0.03, 0.01],
        completions=completions,
    ).write(pipelined, _pipeline_encode_workers=4)

    assert completions != list(range(len(values)))
    assert serial.read_bytes() == pipelined.read_bytes()


def test_pipeline_tiny_write_bound_backpressures_and_stays_identical(
    tmp_path: Path,
    monkeypatch,
):
    values = [torch.full((4096,), index, dtype=torch.uint8)
              for index in range(5)]
    serial = tmp_path / "serial.safetensors"
    pipelined = tmp_path / "pipelined.safetensors"

    monkeypatch.setenv("PRISMAQUANT_EXPORT_PIPELINE", "0")
    _fake_writer(values).write(serial)

    monkeypatch.setenv("PRISMAQUANT_EXPORT_PIPELINE", "1")
    monkeypatch.setenv("PRISMAQUANT_EXPORT_PREFETCH_DEPTH", "2")
    monkeypatch.setenv("PRISMAQUANT_EXPORT_WRITE_QUEUE_BYTES", "4096")
    timings = _fake_writer(
        values,
        delays=[0.08, 0.0, 0.0, 0.0, 0.0],
    ).write(pipelined, _pipeline_encode_workers=2)

    assert timings is not None
    assert timings["backpressure_stalls"] >= 1
    assert serial.read_bytes() == pipelined.read_bytes()


def test_pipeline_encoder_error_removes_temp_and_never_finalizes(
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "failed.safetensors"
    monkeypatch.setenv("PRISMAQUANT_EXPORT_PIPELINE", "1")

    with pytest.raises(RuntimeError, match="fake encoder failed for tensor 1"):
        _fake_writer(
            [torch.ones(8), torch.ones(8), torch.ones(8)],
            fail_index=1,
        ).write(output)

    assert not output.exists()
    assert not (tmp_path / ".failed.safetensors.tmp").exists()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="deferred real-GPU streaming export identity gate",
)
def test_gpu_streaming_export_pipeline_identity(
    tmp_path: Path,
    monkeypatch,
):
    torch.manual_seed(29)
    model = tmp_path / "model"
    model.mkdir()
    qnames = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    tensors = {
        f"{qname}.weight":
            (torch.randn(8, 256) * 0.25).to(torch.bfloat16)
        for qname in qnames
    }
    tensors["model.norm.weight"] = torch.ones(256, dtype=torch.bfloat16)
    save_file(tensors, str(model / "model.safetensors"))
    (model / "config.json").write_text(json.dumps({"hidden_size": 256}))
    assignment = tmp_path / "assignment.json"
    assignment.write_text(json.dumps({qname: "NVFP4_CB_K16"
                                      for qname in qnames}))
    col_weights = {qname: torch.linspace(0.25, 1.25, 256)
                   for qname in qnames}

    serial = tmp_path / "serial"
    monkeypatch.setenv("PRISMAQUANT_EXPORT_PIPELINE", "0")
    export_nvfp4_cb_streaming(
        model,
        assignment,
        serial,
        col_weights,
        device="cuda",
        allow_unstamped_research=True,
    )

    pipelined = tmp_path / "pipelined"
    monkeypatch.setenv("PRISMAQUANT_EXPORT_PIPELINE", "1")
    export_nvfp4_cb_streaming(
        model,
        assignment,
        pipelined,
        col_weights,
        device="cuda",
        allow_unstamped_research=True,
    )

    serial_tree = _tree_bytes(serial)
    pipelined_tree = _tree_bytes(pipelined)
    # The value-bearing artifact must be byte-identical.  The refusal record
    # intentionally binds its own final publication path and caches each
    # freshly written weight file's mtime/ctime, so two independently exported
    # directories cannot have byte-identical shipcards.
    serial_card_bytes = serial_tree.pop("shipcard.json")
    pipelined_card_bytes = pipelined_tree.pop("shipcard.json")
    assert serial_tree == pipelined_tree
    assert len(serial_card_bytes) == len(pipelined_card_bytes)

    serial_card = load_shipcard(serial / "shipcard.json")
    pipelined_card = load_shipcard(pipelined / "shipcard.json")
    # Read-traffic provenance names the staging root whose safetensors headers
    # were inspected.  Like the shipcard's publication-path refusal record,
    # that path must differ between two independent transactions even when the
    # measured traffic and every value-bearing byte are identical.
    serial_read_source = serial_card["build"]["read_gb_per_token"].pop("source")
    pipelined_read_source = (
        pipelined_card["build"]["read_gb_per_token"].pop("source")
    )
    assert ".serial.tmp-" in serial_read_source
    assert ".pipelined.tmp-" in pipelined_read_source
    for key in ("model_sha", "artifact_bytes", "reserved_file_bytes", "build", "slots"):
        assert serial_card[key] == pipelined_card[key]
    assert serial_card["model_dir"] == str(serial)
    assert pipelined_card["model_dir"] == str(pipelined)
