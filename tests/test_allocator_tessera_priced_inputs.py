"""The allocation carries what priced it, so the export gate can bind to it.

RobTand/prismaquant#204: the campaign writes ``hessian_capture.pt`` and
``input_scales.safetensors`` and stamps every cost row with the capture's
content digest and the static ``input_global_scale`` the row was scored
under.  The allocator must carry both, for the units it selected, into the
``__prismaquant__`` metadata of ``layer_config.json`` -- the only thing the
export leg reads -- and ``tessera_export_lane.require_priced_export_inputs``
must then accept exactly the campaign's files and refuse any other.  This is
the whole chain, through the real ``allocator.main()``.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_tessera_scope_endpoints import (  # noqa: E402
    DENSE, EXPERT, SHARED, _allocator_inputs, _cli_scope, _v5_contract,
)

from prismaquant import tessera_export_lane as export  # noqa: E402
from prismaquant.nvfp4_activation_contract import (  # noqa: E402
    input_global_scale_tensor,
)

TRIPLE = {"text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64,
          "fit_tokens": 4096}
SCALE = float(input_global_scale_tensor(448.0 * 6.0 / 37.5).item())
UNITS = (DENSE, EXPERT, SHARED)


def _campaign_outputs(tmp_path, *, hessians=None, scale=SCALE):
    """What the campaign leaves beside its --cache-dir: the two files and the
    digest it stamps on every row."""
    from prismaquant.tessera_campaign import write_export_inputs

    cache = tmp_path / "tessera-cache"
    cache.mkdir(exist_ok=True)
    return write_export_inputs(
        cache,
        hessians=hessians or {name: torch.eye(4) for name in UNITS},
        hessian_rows={name: 4 for name in UNITS},
        hessian_identity=dict(TRIPLE),
        static_scales={name: scale for name in UNITS},
        static_scale_policy="legacy_6_over_calibration_amax.v1")


def _stamp_rows(argv, *, fmt, capture_sha256, scale):
    """Give the fixture cost table what the campaign's rows carry."""
    costs_path = Path(argv[argv.index("--costs") + 1])
    payload = pickle.loads(costs_path.read_bytes())
    for rows in payload["costs"].values():
        rows[fmt]["hessian_identity"] = {
            "supplied": True, **TRIPLE, "capture_sha256": capture_sha256}
        if scale is not None:
            rows[fmt]["input_global_scale"] = scale
    costs_path.write_bytes(pickle.dumps(payload))


def _allocate(tmp_path, monkeypatch, *, fmt, capture_sha256, scale=SCALE):
    from prismaquant import allocator

    _v5_contract(monkeypatch)
    argv = _allocator_inputs(tmp_path, fmt)
    _stamp_rows(argv, fmt=fmt, capture_sha256=capture_sha256, scale=scale)
    monkeypatch.setattr(sys, "argv", [
        "allocator", *argv, "--no-fused-aggregation",
        "--no-packed-aggregation", *_cli_scope()])
    allocator.main()
    return tmp_path / "layer.json"


def test_the_allocation_carries_the_digest_and_the_priced_scales(
        tmp_path, monkeypatch):
    capture, scales, digest = _campaign_outputs(tmp_path)
    layer = _allocate(tmp_path, monkeypatch, fmt="TESSERA_E2M1_K2_R896",
                      capture_sha256=digest)
    metadata = json.loads(layer.read_text())["__prismaquant__"]
    assert metadata["tessera_hessian"]["capture_sha256"] == digest
    assert metadata["tessera_hessian"]["supplied"] is True
    assert metadata["tessera_activation_static_scales"] == {
        "schema": export.PRICED_STATIC_SCALES_SCHEMA,
        "units": {name: SCALE for name in UNITS}}
    # The gate closes the loop: the campaign's own files bind and pass ...
    report = export.require_priced_export_inputs(
        layer, hessian_path=capture, input_scales_path=scales)
    assert report["hessian_capture_sha256"] == digest
    assert report["w4a4_units"] == report["input_scales_bound_units"] == 3


def test_the_gate_refuses_other_files_against_that_allocation(
        tmp_path, monkeypatch):
    """... and anything that is not those files is refused by name, whatever
    the sidecar or the key roster says."""
    capture, scales, digest = _campaign_outputs(tmp_path)
    layer = _allocate(tmp_path, monkeypatch, fmt="TESSERA_E2M1_K2_R896",
                      capture_sha256=digest)
    other = tmp_path / "other"
    other.mkdir()
    other_capture, other_scales, _ = _campaign_outputs(
        other, hessians={name: 2 * torch.eye(4) for name in UNITS},
        scale=10000.0)
    with pytest.raises(export.TesseraExportLaneError,
                       match="capture_sha256.*identity triple agrees"):
        export.require_priced_export_inputs(
            layer, hessian_path=other_capture, input_scales_path=scales)
    with pytest.raises(export.TesseraExportLaneError,
                       match="input_global_scale = 10000.0 but the "
                             "allocation priced"):
        export.require_priced_export_inputs(
            layer, hessian_path=capture, input_scales_path=other_scales)


def test_a_table_whose_rows_priced_no_scale_leaves_the_unit_unbound(
        tmp_path, monkeypatch):
    """No default is invented for a W4A4 row that never said what priced it;
    the gate refuses that unit by name."""
    capture, scales, digest = _campaign_outputs(tmp_path)
    layer = _allocate(tmp_path, monkeypatch, fmt="TESSERA_E2M1_K2_R896",
                      capture_sha256=digest, scale=None)
    metadata = json.loads(layer.read_text())["__prismaquant__"]
    assert metadata["tessera_activation_static_scales"]["units"] == {}
    with pytest.raises(export.TesseraExportLaneError,
                       match="priced no input_global_scale for"):
        export.require_priced_export_inputs(
            layer, hessian_path=capture, input_scales_path=scales)


def test_a_dense_e4m3_selection_prices_no_static_scale(tmp_path, monkeypatch):
    """A route that does not execute the static contract carries no scale on
    its row, and the block says so (empty, present) rather than being
    absent."""
    capture, _scales, digest = _campaign_outputs(tmp_path)
    layer = _allocate(tmp_path, monkeypatch, fmt="TESSERA_E4M3_K1_R1024",
                      capture_sha256=digest, scale=None)
    metadata = json.loads(layer.read_text())["__prismaquant__"]
    assert metadata["tessera_activation_static_scales"] == {
        "schema": export.PRICED_STATIC_SCALES_SCHEMA, "units": {}}
    report = export.require_priced_export_inputs(layer, hessian_path=capture)
    assert report["input_scales_required"] is False


def test_a_stock_allocation_is_byte_identical(tmp_path, monkeypatch):
    from prismaquant import allocator

    monkeypatch.setattr(sys, "argv", [
        "allocator", *_allocator_inputs(tmp_path, "BF16"),
        "--no-fused-aggregation", "--no-packed-aggregation", *_cli_scope()])
    allocator.main()
    metadata = json.loads((tmp_path / "layer.json").read_text())["__prismaquant__"]
    assert "tessera_activation_static_scales" not in metadata
    assert "tessera_hessian" not in metadata
