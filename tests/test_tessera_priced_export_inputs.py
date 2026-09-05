"""The export leg receives the inputs the allocation was priced under.

The campaign prices every rung against a specific Hessian capture and, on the
W4A4 routes, a specific static ``input_global_scale`` per unit.  Tessera's
exporter takes both (``--hessian``, ``--input-scales``) -- and encodes
weights-only without complaint when the first is omitted, and refuses NVFP4
routes only after everything else is encoded when the second is.  So the
producer's arm must (a) hand them through and (b) fail closed BEFORE the
encode when the allocation declares a priced requirement the supplied inputs
do not satisfy (RobTand/prismaquant#193).
"""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from prismaquant import tessera_export_lane as export

ROOT = Path(__file__).resolve().parents[1]
DENSE_E4M3 = "model.layers.0.self_attn.o_proj"
DENSE_E2M1 = "model.layers.0.mlp.up_proj"

TRIPLE = {"text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64,
          "fit_tokens": 4096}


def _assignment(tmp_path, *, formats, hessian_block="modern-supplied"):
    payload = {name: {"data_type": "tessera", "bits": 4,
                      "tessera_format": fmt}
               for name, fmt in formats.items()}
    blocks = {
        "modern-supplied": {"supplied": True, **TRIPLE},
        "modern-weights-only": {"supplied": False, **TRIPLE},
        "legacy-supplied": {"supplied": True, "text_sha": "abc",
                            "token_count": 4096},
        None: None,
    }
    meta = {}
    if blocks[hessian_block] is not None:
        meta["tessera_hessian"] = blocks[hessian_block]
    payload["__prismaquant__"] = meta
    path = tmp_path / "layer_config.json"
    path.write_text(json.dumps({
        name: value for name, value in payload.items()}))
    return path


def _capture(tmp_path, identity=None, *, sidecar=True, role="fit"):
    from prismaquant.tessera_campaign import write_export_inputs

    identity = dict(TRIPLE if identity is None else identity)
    hessians = {DENSE_E4M3: torch.eye(4)}
    capture, _scales = write_export_inputs(
        tmp_path, hessians=hessians, hessian_rows={DENSE_E4M3: 4},
        hessian_identity=identity, static_scales={}, static_scale_policy="x")
    if role != "fit":
        payload = torch.load(capture, weights_only=False)
        payload["provenance"]["hessian_role"] = role
        torch.save(payload, capture)
        Path(str(capture) + ".provenance.json").write_text(
            json.dumps(payload["provenance"]))
    if not sidecar:
        Path(str(capture) + ".provenance.json").unlink()
    return capture


def _scales_file(tmp_path, units):
    from prismaquant.tessera_campaign import write_export_inputs

    _capture_path, scales = write_export_inputs(
        tmp_path, hessians=None, hessian_rows={}, hessian_identity={},
        static_scales={name: 448.0 * 6.0 / 37.5 for name in units},
        static_scale_policy="legacy_6_over_calibration_amax.v1")
    return scales


# ---------------------------------------------------------------------------
# The Hessian half
# ---------------------------------------------------------------------------

def test_an_h_aware_allocation_with_no_capture_refuses(tmp_path):
    """The exporter would encode weights-only and raise nothing.

    ``export_tessera_serving.py`` defaults ``--hessian`` to None and builds an
    ActivationSource only when it is present, so the omission ships different
    bytes at the same format name -- silently. The producer's gate is the
    refusal.
    """
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    with pytest.raises(export.TesseraExportLaneError,
                       match="priced H-aware.*no --hessian"):
        export.require_priced_export_inputs(assignment)


def test_a_capture_with_the_wrong_identity_refuses(tmp_path):
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    capture = _capture(tmp_path, {**TRIPLE, "fit_ids_sha256": "c" * 64})
    with pytest.raises(export.TesseraExportLaneError,
                       match="not the capture that priced"):
        export.require_priced_export_inputs(assignment, hessian_path=capture)


def test_the_campaigns_own_capture_binds_and_passes(tmp_path):
    """write_export_inputs -> the gate, end to end: the sidecar identity the
    campaign writes is exactly what the allocation's #195 stamp compares."""
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    capture = _capture(tmp_path)
    report = export.require_priced_export_inputs(
        assignment, hessian_path=capture)
    assert report["hessian_required"] is True
    assert report["hessian"] == str(capture)
    assert report["input_scales_required"] is False


def test_a_capture_without_a_sidecar_is_read_from_the_payload(tmp_path):
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    capture = _capture(tmp_path, sidecar=False)
    report = export.require_priced_export_inputs(
        assignment, hessian_path=capture)
    assert report["hessian"] == str(capture)


def test_a_held_out_capture_must_not_shape_bytes(tmp_path):
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    capture = _capture(tmp_path, role="held-out")
    with pytest.raises(export.TesseraExportLaneError,
                       match="held-out.*must not shape bytes"):
        export.require_priced_export_inputs(assignment, hessian_path=capture)


def test_a_weights_only_allocation_refuses_a_stray_capture(tmp_path):
    """The same drift in the other direction: H-aware bytes under a
    weights-only price are also not the artifact priced."""
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"},
        hessian_block="modern-weights-only")
    capture = _capture(tmp_path)
    with pytest.raises(export.TesseraExportLaneError,
                       match="priced weights-only.*--hessian"):
        export.require_priced_export_inputs(assignment, hessian_path=capture)


def test_an_undeclared_allocation_fails_closed(tmp_path):
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"},
        hessian_block=None)
    with pytest.raises(export.TesseraExportLaneError,
                       match="no tessera_hessian pricing state"):
        export.require_priced_export_inputs(assignment)


def test_a_pre_triple_allocation_cannot_be_bound(tmp_path):
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"},
        hessian_block="legacy-supplied")
    capture = _capture(tmp_path)
    with pytest.raises(export.TesseraExportLaneError,
                       match="no required identity triple"):
        export.require_priced_export_inputs(assignment, hessian_path=capture)


def test_an_allocation_with_no_tessera_units_needs_nothing(tmp_path):
    path = tmp_path / "layer_config.json"
    path.write_text(json.dumps({"model.layers.0.mlp.up_proj": "BF16"}))
    report = export.require_priced_export_inputs(path)
    assert report == {"hessian_required": False, "hessian": None,
                      "input_scales_required": False, "input_scales": None,
                      "w4a4_units": 0}


def test_the_priced_input_triple_matches_tesseras_roster():
    """The gate's spelled triple and Tessera's derived one cannot drift."""
    from prismaquant.tessera_hessian import HESSIAN_IDENTITY_FIELDS

    assert set(export.PRICED_HESSIAN_IDENTITY_FIELDS) == set(
        HESSIAN_IDENTITY_FIELDS)


# ---------------------------------------------------------------------------
# The static-activation-scale half
# ---------------------------------------------------------------------------

def test_w4a4_selection_requires_input_scales(tmp_path):
    """The exporter refuses NVFP4 routes without scales -- after encoding
    everything else. This refusal is before a single unit is encoded."""
    assignment = _assignment(
        tmp_path, formats={DENSE_E2M1: "TESSERA_E2M1_K2_R896"},
        hessian_block="modern-weights-only")
    with pytest.raises(export.TesseraExportLaneError,
                       match="static NVFP4 activation contract.*--input-scales"):
        export.require_priced_export_inputs(assignment)


def test_input_scales_must_cover_every_selected_w4a4_unit(tmp_path):
    assignment = _assignment(
        tmp_path, formats={DENSE_E2M1: "TESSERA_E2M1_K2_R896"},
        hessian_block="modern-weights-only")
    scales = _scales_file(tmp_path, ["model.layers.9.some_other_unit"])
    with pytest.raises(export.TesseraExportLaneError,
                       match="carries no input_global_scale"):
        export.require_priced_export_inputs(
            assignment, input_scales_path=scales)


def test_the_campaigns_own_scales_file_covers_and_passes(tmp_path):
    assignment = _assignment(
        tmp_path, formats={DENSE_E2M1: "TESSERA_E2M1_K2_R896",
                           DENSE_E4M3: "TESSERA_E4M3_K1_R1024"},
        hessian_block="modern-weights-only")
    scales = _scales_file(tmp_path, [DENSE_E2M1])
    report = export.require_priced_export_inputs(
        assignment, input_scales_path=scales)
    assert report["input_scales_required"] is True
    assert report["w4a4_units"] == 1
    assert report["input_scales"] == str(scales)


def test_dense_e4m3_selection_needs_no_scales(tmp_path):
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"},
        hessian_block="modern-weights-only")
    report = export.require_priced_export_inputs(assignment)
    assert report["input_scales_required"] is False
    assert report["w4a4_units"] == 0


# ---------------------------------------------------------------------------
# The driver's arm
# ---------------------------------------------------------------------------

def test_the_shipping_arm_hands_the_priced_inputs_to_the_exporter():
    """run-pipeline.sh's Tessera arm forwarded only --plan-json and --device.

    The pre-#193 invocation supplied neither --hessian nor --input-scales, so
    an H-aware allocation re-encoded weights-only and an E2M1 allocation died
    inside the exporter after the plan translation. Both flags now ride one
    array that the preflight validates first.
    """
    driver = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    exporter = driver.index("experiments/export_tessera_serving.py")
    invocation = driver[exporter:driver.index("2>&1 | tee", exporter)]
    assert '"${TESSERA_PRICED_INPUT_ARGS[@]}"' in invocation
    translator = driver.index(
        'python3 "${TESSERA_REPO%/}/experiments/plan_from_layer_config.py"')
    preflight = driver.rfind(
        "python3 -m prismaquant.tessera_export_lane", 0, translator)
    gate = driver[preflight:driver.index("; then", preflight)]
    assert '--assignment "${WORK_DIR}/artifacts/layer_config.json"' in gate
    assert '"${TESSERA_PRICED_INPUT_ARGS[@]}"' in gate
    # Built inside the arm, before the preflight consumes it -- the refusal
    # must fire before the plan translation, let alone the encode.
    built = driver.index("TESSERA_PRICED_INPUT_ARGS+=(--hessian")
    assert built < preflight
    for knob in ('"${TESSERA_HESSIAN:=}"', '"${TESSERA_INPUT_SCALES:=}"'):
        assert knob in driver, knob


def test_cli_passes_priced_inputs_to_preflight(tmp_path, monkeypatch):
    calls = []

    def preflight(model, **kwargs):
        calls.append(kwargs)
        raise export.TesseraExportLaneError("fixture stop after boundary")

    monkeypatch.setattr(export, "preflight", preflight)
    assert export.main([
        "--model", str(tmp_path), "--assignment", str(tmp_path / "l.json"),
        "--hessian", str(tmp_path / "h.pt"),
        "--input-scales", str(tmp_path / "s.safetensors"),
        "--tessera-platform", "sm_121",
        "--tessera-runtime-image", "example/runtime@sha256:" + "a" * 64,
        "--tessera-execution-mode", "eager", "--tessera-residency", "resident",
    ]) == 2
    assert calls[0]["hessian_path"] == str(tmp_path / "h.pt")
    assert calls[0]["input_scales_path"] == str(tmp_path / "s.safetensors")


def test_priced_inputs_without_an_assignment_are_refused(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert export.main([
        "--model", str(tmp_path), "--hessian", str(tmp_path / "h.pt"),
    ]) == 2
