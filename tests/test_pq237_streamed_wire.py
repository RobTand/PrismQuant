"""Retain and refuse real tiny CPU wire artifacts; no model or serving claim.

The 32x32 E4M3 fixture follows Tessera's test_cached_producer.py. Its encoder,
receipt validator, and bytes-only decoder are real; only disk corruption is
injected. The fixture deliberately encodes without a calibration Hessian.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.pq237_joint_aura_streamed import retain_production_wire


QNAME = "model.layers.0.mlp.down_proj"
FORMAT = "TESSERA_E4M3_K1_R1024"


@pytest.fixture(scope="module")
def encoded_wire():
    from tessera.alphabet import E4M3_GRID
    from tessera.export import encode_linear
    from tessera.unit_artifact import read_unit_artifact

    weight = torch.randn(32, 32, generator=torch.Generator().manual_seed(183)).bfloat16()
    unit = encode_linear(weight, grid=E4M3_GRID, q256=1024,
                         name=FORMAT, verify=False)
    rendered = read_unit_artifact(unit.blob, device="cpu").to(weight.dtype)
    return weight, rendered, unit.blob


@pytest.fixture
def wire_dir(tmp_path):
    directory = tmp_path / "wire"
    directory.mkdir()
    return directory


def test_retained_wire_keeps_original_bytes_and_reusable_producer_record(encoded_wire, wire_dir):
    from tessera import cached_unit
    from tessera.alphabet import E4M3_GRID
    from tessera.unit_artifact import read_unit_artifact

    weight, rendered, blob = encoded_wire
    receipt = retain_production_wire(
        weight, rendered, blob, qname=QNAME, fmt=FORMAT,
        activation_source=None, wire_dir=wire_dir)
    # JSON is the downstream receipt boundary; the original input objects are
    # independently available to verify the record read from that boundary.
    receipt = json.loads(json.dumps(receipt))
    path = wire_dir.parent / receipt["wire_path"]
    assert path.parent == wire_dir
    assert path.read_bytes() == blob
    assert receipt["blob_bytes"] == len(blob)
    assert receipt["blob_sha256"] == hashlib.sha256(blob).hexdigest()
    assert receipt["wire_record"]["file"] == path.name
    assert set(wire_dir.iterdir()) == {path}

    expected = cached_unit.encoding_input_identity(
        weight, QNAME, E4M3_GRID, 1024, activation=None)
    accepted = cached_unit.verify_cached_unit(path.read_bytes(), receipt["wire_record"], expected)
    assert accepted.blob == blob
    assert receipt["wire_record"]["identity"]["calibration"] is None
    decoded = read_unit_artifact(accepted.blob, device="cpu").to(rendered.dtype)
    assert torch.equal(decoded, rendered)
    assert receipt["wire_decoded_weight"]["shape"] == list(rendered.shape)
    assert receipt["wire_decoded_weight"]["dtype"] == str(rendered.dtype)
    assert receipt["wire_decoded_weight"]["content_sha256"] == hashlib.sha256(
        decoded.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def test_retention_refuses_corruption_of_persisted_bytes(encoded_wire, wire_dir, monkeypatch):
    weight, rendered, blob = encoded_wire
    write_bytes = Path.write_bytes
    corrupted = []

    def corrupt_write(path, data):
        if path.parent == wire_dir:
            changed = bytearray(data)
            changed[-1] ^= 1
            data = bytes(changed)
            corrupted.append(data)
        return write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", corrupt_write)
    with pytest.raises(ValueError, match="blob size/sha256 mismatch"):
        retain_production_wire(weight, rendered, blob, qname=QNAME, fmt=FORMAT,
                               activation_source=None, wire_dir=wire_dir)
    assert corrupted == [blob[:-1] + bytes([blob[-1] ^ 1])]
    persisted = list(wire_dir.glob("*.tessera"))
    assert len(persisted) == 1
    assert persisted[0].read_bytes() == corrupted[0]


@pytest.mark.parametrize("mismatch", ["values", "shape"])
def test_retention_refuses_wire_decode_that_differs_from_cached_render(
        encoded_wire, wire_dir, mismatch):
    weight, rendered, blob = encoded_wire
    if mismatch == "values":
        wrong = rendered.clone()
        wrong[0, 0] += 1
    else:
        wrong = rendered.reshape(16, 64)
    with pytest.raises(ValueError, match="does not decode to the production cache tensor"):
        retain_production_wire(weight, wrong, blob, qname=QNAME, fmt=FORMAT,
                               activation_source=None, wire_dir=wire_dir)
