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
from prismaquant.nvfp4_activation_contract import input_global_scale_tensor

ROOT = Path(__file__).resolve().parents[1]
DENSE_E4M3 = "model.layers.0.self_attn.o_proj"
DENSE_E2M1 = "model.layers.0.mlp.up_proj"

TRIPLE = {"text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64,
          "fit_tokens": 4096}
#: The capture the fixtures below write, so an assignment can be stamped with
#: the digest of exactly that payload (what the campaign's cost rows carry).
HESSIANS = {DENSE_E4M3: torch.eye(4)}


def _digest(hessians=None, identity=None):
    return export.hessian_capture_sha256(
        HESSIANS if hessians is None else hessians,
        {**TRIPLE, "hessian_role": "fit"} if identity is None else identity)


def _assignment(tmp_path, *, formats, hessian_block="modern-supplied",
                capture_sha256="own", scales="own"):
    """``capture_sha256``: ``"own"`` stamps the digest of the fixture capture,
    ``None`` leaves the (pre-#204) block without one, any other string is
    stamped verbatim.  ``scales``: ``"own"`` stamps the fixture scale for every
    W4A4 unit, ``None`` omits the (pre-#204) block, a mapping is stamped."""
    payload = {name: {"data_type": "tessera", "bits": 4,
                      "tessera_format": fmt}
               for name, fmt in formats.items()}
    stamp = {} if capture_sha256 is None else {
        "capture_sha256": _digest() if capture_sha256 == "own"
        else capture_sha256}
    blocks = {
        "modern-supplied": {"supplied": True, **TRIPLE, **stamp},
        "modern-weights-only": {"supplied": False, **TRIPLE},
        "legacy-supplied": {"supplied": True, "text_sha": "abc",
                            "token_count": 4096},
        None: None,
    }
    meta = {}
    if blocks[hessian_block] is not None:
        meta["tessera_hessian"] = blocks[hessian_block]
    if scales is not None:
        from prismaquant.tessera_campaign import _format_executes_static_nvfp4

        units = ({name: SCALE for name, fmt in formats.items()
                  if _format_executes_static_nvfp4(fmt)}
                 if scales == "own" else dict(scales))
        meta["tessera_activation_static_scales"] = {
            "schema": export.PRICED_STATIC_SCALES_SCHEMA, "units": units}
    payload["__prismaquant__"] = meta
    path = tmp_path / "layer_config.json"
    path.write_text(json.dumps({
        name: value for name, value in payload.items()}))
    return path


def _capture(tmp_path, identity=None, *, sidecar=True, role="fit",
             hessians=None):
    from prismaquant.tessera_campaign import write_export_inputs

    identity = dict(TRIPLE if identity is None else identity)
    hessians = dict(HESSIANS if hessians is None else hessians)
    capture, _scales, _digest = write_export_inputs(
        tmp_path, hessians=hessians, hessian_rows={DENSE_E4M3: 4},
        hessian_identity=identity, static_scales={}, static_scale_policy="x")
    if role != "fit":
        payload = torch.load(capture, weights_only=False)
        payload["provenance"]["hessian_role"] = role
        torch.save(payload, capture)
        Path(str(capture) + ".provenance.json").write_text(json.dumps({
            **payload["provenance"],
            "capture_sha256": export.hessian_capture_sha256(
                payload["H"], payload["provenance"])}))
    if not sidecar:
        Path(str(capture) + ".provenance.json").unlink()
    return capture


#: The static scale the fixture files carry per unit, F32-rounded the way the
#: campaign's ``input_global_scale_from_max_abs`` rounds the value it prices.
SCALE = float(input_global_scale_tensor(448.0 * 6.0 / 37.5).item())


def _scales_file(tmp_path, units, value=SCALE):
    from prismaquant.tessera_campaign import write_export_inputs

    _capture_path, scales, _digest = write_export_inputs(
        tmp_path, hessians=None, hessian_rows={}, hessian_identity={},
        static_scales={name: value for name in units},
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
    """write_export_inputs -> the gate, end to end: the payload the campaign
    writes is exactly what the allocation's stamp (#195 triple, #204 digest)
    compares."""
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    capture = _capture(tmp_path)
    report = export.require_priced_export_inputs(
        assignment, hessian_path=capture)
    assert report["hessian_required"] is True
    assert report["hessian"] == str(capture)
    assert report["hessian_capture_sha256"] == _digest()
    assert report["input_scales_required"] is False


def test_a_capture_without_a_sidecar_is_read_from_the_payload(tmp_path):
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    capture = _capture(tmp_path, sidecar=False)
    report = export.require_priced_export_inputs(
        assignment, hessian_path=capture)
    assert report["hessian"] == str(capture)


# -- #204: the identity is bound to the .pt payload Tessera reads, by content.

def _rewrite_payload(capture, *, identity=None, hessians=None):
    """Replace the ``.pt`` the exporter will load, leaving the sidecar alone --
    the on-disk state the pre-#204 gate could not see."""
    payload = torch.load(capture, weights_only=False)
    if identity is not None:
        payload["provenance"] = {**dict(identity), "hessian_role": "fit"}
    if hessians is not None:
        payload["H"] = dict(hessians)
    torch.save(payload, capture)
    return payload


def test_a_stale_sidecar_over_a_rewritten_payload_refuses(tmp_path):
    """codex ``prismaquant_seam_inputs.py`` case 1: sidecar A, payload B.

    Tessera's ``ActivationSource.from_capture`` never reads the sidecar, so
    the pre-#204 gate compared the allocation against a file the encode
    ignores and accepted a payload carrying a different draw AND a different
    H.  The sidecar now carries the payload's digest and must seal the payload
    beside it; a sidecar that seals something else is refused by name.
    """
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    capture = _capture(tmp_path)
    _rewrite_payload(capture, identity={**TRIPLE, "fit_ids_sha256": "c" * 64},
                     hessians={DENSE_E4M3: 2 * torch.eye(4)})
    with pytest.raises(export.TesseraExportLaneError,
                       match="provenance.json.*does not seal the payload"):
        export.require_priced_export_inputs(assignment, hessian_path=capture)


def test_an_unsealed_pre_204_sidecar_refuses(tmp_path):
    """A sidecar written before the seal existed carries no digest; it is not
    evidence of anything and is refused rather than read."""
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    capture = _capture(tmp_path)
    Path(str(capture) + ".provenance.json").write_text(json.dumps(
        {**TRIPLE, "hessian_role": "fit"}))
    with pytest.raises(export.TesseraExportLaneError,
                       match="provenance.json.*does not seal the payload"):
        export.require_priced_export_inputs(assignment, hessian_path=capture)


def test_the_same_triple_over_a_different_hessian_refuses(tmp_path):
    """codex case 2: H = I vs H = 2I under one identity triple.

    The triple names the token draw, not the Hessian content; two captures
    of one draw with different H encode different bytes at the same format
    name.  The allocation carries the digest of the capture that priced it,
    and a capture whose content digests differently is refused by name even
    when every triple field agrees.
    """
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    capture = _capture(tmp_path, hessians={DENSE_E4M3: 2 * torch.eye(4)},
                       sidecar=False)
    with pytest.raises(export.TesseraExportLaneError,
                       match="capture_sha256.*identity triple agrees"):
        export.require_priced_export_inputs(assignment, hessian_path=capture)


def test_a_matched_capture_is_accepted_with_or_without_its_sidecar(tmp_path):
    """The accepted case beside the two refusals above: the payload the
    campaign wrote, digest-equal to the allocation's stamp."""
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    capture = _capture(tmp_path)
    for sidecar in (True, False):
        if not sidecar:
            Path(str(capture) + ".provenance.json").unlink()
        report = export.require_priced_export_inputs(
            assignment, hessian_path=capture)
        assert report["hessian_capture_sha256"] == _digest()


def test_an_allocation_without_a_capture_digest_is_unbound(tmp_path):
    """A pre-#204 allocation carries the triple and no digest.  It cannot be
    bound to a payload, and 'unbound' is refused by name -- never read as
    'matches any capture with this triple' (principle 2)."""
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"},
        capture_sha256=None)
    capture = _capture(tmp_path)
    with pytest.raises(export.TesseraExportLaneError,
                       match="no capture_sha256.*unbound"):
        export.require_priced_export_inputs(assignment, hessian_path=capture)


def test_a_payload_without_hessians_cannot_be_bound(tmp_path):
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    capture = tmp_path / "hessian_capture.pt"
    torch.save({"provenance": {**TRIPLE, "hessian_role": "fit"}}, capture)
    with pytest.raises(export.TesseraExportLaneError,
                       match="carries no 'H' mapping"):
        export.require_priced_export_inputs(assignment, hessian_path=capture)


def test_the_digest_rule_covers_content_and_capture_context():
    """One documented rule, mirroring Tessera's ``capture_sha256``: the
    identity triple, the capture context (model, seqlen, source) and every
    unit's H by dtype, shape and contiguous bytes.  Nothing else."""
    base = {**TRIPLE, "hessian_role": "fit", "model": "m", "seqlen": 2048,
            "source": "wikitext", "seed": 0, "nsamples": 8}
    ours = export.hessian_capture_sha256(HESSIANS, base)
    assert ours == export.hessian_capture_sha256(
        {DENSE_E4M3: torch.eye(4).clone()}, {**base, "seed": 1, "nsamples": 9,
                                             "path": "/elsewhere"})
    assert ours != export.hessian_capture_sha256(
        {DENSE_E4M3: 2 * torch.eye(4)}, base)
    assert ours != export.hessian_capture_sha256(
        {DENSE_E4M3: torch.eye(4, dtype=torch.float64)}, base)
    assert ours != export.hessian_capture_sha256(
        {DENSE_E4M3: torch.eye(4), "other": torch.eye(2)}, base)
    for field, value in (("model", "n"), ("seqlen", 4096), ("source", "c4"),
                         ("fit_tokens", 4097)):
        assert ours != export.hessian_capture_sha256(
            HESSIANS, {**base, field: value}), field


def test_the_digest_rule_is_tesseras_capture_seal(tmp_path):
    """Where the pinned Tessera publishes ``capture_sha256`` (release
    e78959e+), the producer's rule must reproduce it byte for byte and the
    gate cross-checks the two on every bind; at a pin that predates the seal
    the gate reports the cross-check as unavailable.

    This is the *runtime* half of the drift guard and it can only run where
    the seal exists.  The roster half needs no ``capture_sha256`` and runs at
    every pin: ``test_the_capture_context_roster_is_tesseras`` below.
    """
    tessera_export = pytest.importorskip("tessera.export")
    source_cls = tessera_export.ActivationSource
    if not hasattr(source_cls, "capture_sha256"):
        pytest.skip("pinned tessera has no ActivationSource.capture_sha256")
    provenance = {**TRIPLE, "hessian_role": "fit", "model": "m",
                  "seqlen": 2048, "source": "wikitext"}
    source = source_cls(hessians=dict(HESSIANS), provenance=dict(provenance))
    assert export.hessian_capture_sha256(HESSIANS, provenance) == \
        source.capture_sha256()
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    report = export.require_priced_export_inputs(
        assignment, hessian_path=_capture(tmp_path))
    assert report["hessian_capture_seal_crosscheck"] == \
        "tessera.export.ActivationSource.capture_sha256"


def test_a_drift_between_the_two_seal_rules_refuses(tmp_path, monkeypatch):
    """If Tessera's seal and ours ever disagree on one payload, nothing binds
    until they agree -- refused by name, not resolved in either's favour."""
    monkeypatch.setattr(export, "_tessera_capture_seal",
                        lambda hessians, provenance: "f" * 64)
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    with pytest.raises(export.TesseraExportLaneError,
                       match="disagrees with tessera.export"):
        export.require_priced_export_inputs(
            assignment, hessian_path=_capture(tmp_path))


def test_the_crosscheck_is_reported_unavailable_at_a_pre_seal_pin(tmp_path):
    tessera_export = pytest.importorskip("tessera.export")
    if hasattr(tessera_export.ActivationSource, "capture_sha256"):
        pytest.skip("this tessera publishes the seal; see the test above")
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    report = export.require_priced_export_inputs(
        assignment, hessian_path=_capture(tmp_path))
    assert report["hessian_capture_seal_crosscheck"] is None


def test_the_campaign_seals_the_sidecar_to_the_payload_it_wrote(tmp_path):
    """The sidecar carries the digest of exactly the ``.pt`` beside it, the
    writer returns that digest for the cost rows, and no temp files remain."""
    from prismaquant.tessera_campaign import write_export_inputs

    capture, _scales, digest = write_export_inputs(
        tmp_path, hessians=dict(HESSIANS), hessian_rows={DENSE_E4M3: 4},
        hessian_identity=dict(TRIPLE), static_scales={},
        static_scale_policy="x")
    payload = torch.load(capture, weights_only=False)
    assert digest == export.hessian_capture_sha256(
        payload["H"], payload["provenance"]) == _digest()
    sidecar = json.loads(Path(str(capture) + ".provenance.json").read_text())
    assert sidecar["capture_sha256"] == digest
    assert sidecar["capture_sha256_schema"] == \
        export.HESSIAN_CAPTURE_SHA256_SCHEMA
    assert {k: v for k, v in sidecar.items()
            if not k.startswith("capture_sha256")} == payload["provenance"]
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_the_campaign_never_leaves_a_sidecar_that_disagrees(tmp_path,
                                                            monkeypatch):
    """Payload and sidecar are both staged and renamed, the old sidecar is
    removed before the payload lands: a failure at any point leaves the
    sidecar absent or sealing the payload beside it, never stale."""
    import os

    from prismaquant import tessera_campaign

    first = _capture(tmp_path)
    real_replace = os.replace

    def failing_replace(src, dst):
        if str(dst).endswith(".provenance.json"):
            raise OSError("fixture: disk full at the sidecar rename")
        return real_replace(src, dst)

    monkeypatch.setattr(tessera_campaign.os, "replace", failing_replace)
    with pytest.raises(OSError, match="fixture"):
        tessera_campaign.write_export_inputs(
            tmp_path, hessians={DENSE_E4M3: 2 * torch.eye(4)},
            hessian_rows={DENSE_E4M3: 4}, hessian_identity=dict(TRIPLE),
            static_scales={}, static_scale_policy="x")
    assert not Path(str(first) + ".provenance.json").exists()
    payload = torch.load(first, weights_only=False)
    assert torch.equal(payload["H"][DENSE_E4M3], 2 * torch.eye(4))
    # And the payload that did land binds on its own.
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"},
        capture_sha256=export.hessian_capture_sha256(
            payload["H"], payload["provenance"]))
    export.require_priced_export_inputs(assignment, hessian_path=first)


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
                      "hessian_capture_sha256": None,
                      "hessian_capture_seal_crosscheck": None,
                      "input_scales_required": False, "input_scales": None,
                      "w4a4_units": 0, "input_scales_bound_units": 0}


def test_the_priced_input_triple_matches_tesseras_roster():
    """The gate's spelled triple and Tessera's derived one cannot drift."""
    from prismaquant.tessera_hessian import HESSIAN_IDENTITY_FIELDS

    assert set(export.PRICED_HESSIAN_IDENTITY_FIELDS) == set(
        HESSIAN_IDENTITY_FIELDS)


def test_the_capture_context_roster_is_tesseras():
    """The other half of the same seal's roster, read from its owner.

    Runs at **both** pins, unlike the runtime seal comparison: a tuple
    equality needs no ``ActivationSource.capture_sha256``.

    * At a Tessera that publishes ``CAPTURE_CONTEXT`` (the release tip), the
      gate must have read the constant, not a typed copy of it -- compared as
      an ordered tuple, not a set, because a rename leaves the set alone while
      the digest's ``.get`` starts sealing ``None``.
    * At a Tessera that predates the constant (the pinned dev 1221d2a), the
      gate must be using the ONE documented fallback and saying so, since that
      pin also has no ``capture_sha256`` and nothing else can see the roster.

    RobTand/prismaquant#216: before this, the roster was typed at
    ``tessera_export_lane.py:494`` and nothing read
    ``tessera.export.CAPTURE_CONTEXT`` anywhere in the tree.
    """
    tessera_export = pytest.importorskip("tessera.export")
    published = getattr(tessera_export, "CAPTURE_CONTEXT", None)
    if published is None:
        assert export.CAPTURE_CONTEXT_FIELDS == export._CAPTURE_CONTEXT_FALLBACK
        assert export.CAPTURE_CONTEXT_FIELDS_SOURCE == \
            export._CAPTURE_CONTEXT_FROM_FALLBACK
        assert not hasattr(tessera_export.ActivationSource, "capture_sha256"), \
            "a Tessera that seals but publishes no roster is a new state"
        return
    assert tuple(export.CAPTURE_CONTEXT_FIELDS) == tuple(published)
    assert export.CAPTURE_CONTEXT_FIELDS_SOURCE == \
        export._CAPTURE_CONTEXT_FROM_TESSERA


def _pretend_tessera_publishes(monkeypatch, fields):
    """Run as if the installed Tessera's ``CAPTURE_CONTEXT`` were ``fields``.

    Patched at the module attribute the release tip publishes, so the refusal
    is driven through the gate's real read of Tessera; at a pin that predates
    the constant (1221d2a) there is no such attribute and the gate's own
    reader is patched instead.  Either way ``CAPTURE_CONTEXT_FIELDS`` keeps
    the roster it resolved at import -- which is the drift being staged.
    """
    tessera_export = pytest.importorskip("tessera.export")
    if hasattr(tessera_export, "CAPTURE_CONTEXT"):
        monkeypatch.setattr(tessera_export, "CAPTURE_CONTEXT", tuple(fields))
    else:
        monkeypatch.setattr(export, "_tessera_capture_context",
                            lambda: tuple(fields))


def test_a_capture_context_field_our_roster_lacks_refuses(monkeypatch):
    """Failure state (a) of #216, and it needs no ``capture_sha256``.

    A Tessera whose ``CAPTURE_CONTEXT`` names a field this roster lacks makes
    the two digest rules cover different provenance.  Where the running
    Tessera also predates the seal, ``_crosscheck_capture_seal`` compares
    nothing, so two captures differing only in the new field digest
    identically and ``require_priced_export_inputs`` binds an allocation to a
    capture that did not price it -- #204's failure, reintroduced silently.
    It is now refused by name, before a byte is digested, at every pin.
    """
    _pretend_tessera_publishes(
        monkeypatch, tuple(export.CAPTURE_CONTEXT_FIELDS) + ("layout",))
    with pytest.raises(export.TesseraExportLaneError,
                       match="tessera.export.CAPTURE_CONTEXT names"):
        export.hessian_capture_sha256(HESSIANS, {**TRIPLE,
                                                 "hessian_role": "fit"})


def test_the_roster_refusal_names_both_rosters_and_where_ours_came_from(
        monkeypatch):
    """A refusal an operator can act on: which fields each side names, which
    are missing here, and whether ours was read from Tessera or fell back."""
    _pretend_tessera_publishes(monkeypatch, ("model", "seqlen", "layout"))
    with pytest.raises(export.TesseraExportLaneError) as excinfo:
        export._require_capture_context_roster()
    message = str(excinfo.value)
    assert "missing here: ('layout',)" in message
    assert "not in Tessera: ('source',)" in message
    assert export.CAPTURE_CONTEXT_FIELDS_SOURCE in message


def test_the_gate_refuses_a_bind_under_a_drifted_roster(tmp_path, monkeypatch):
    """The roster refusal reaches the export gate itself, not only the digest
    helper: nothing binds while the two rules cover different provenance."""
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"})
    capture = _capture(tmp_path)
    _pretend_tessera_publishes(
        monkeypatch, tuple(export.CAPTURE_CONTEXT_FIELDS) + ("layout",))
    with pytest.raises(export.TesseraExportLaneError,
                       match="different provenance"):
        export.require_priced_export_inputs(assignment, hessian_path=capture)


def test_a_tessera_without_the_constant_uses_the_documented_fallback(
        monkeypatch):
    """The fallback is reached only through ``_tessera_capture_context``
    returning None, and it is the tuple copied from tessera 3efd690 -- the one
    place the roster is typed."""
    monkeypatch.setattr(export, "_tessera_capture_context", lambda: None)
    fields, source = export._capture_context_fields()
    assert fields == export._CAPTURE_CONTEXT_FALLBACK == (
        "model", "seqlen", "source")
    assert source == export._CAPTURE_CONTEXT_FROM_FALLBACK
    assert "3efd690" in source


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
    assert report["input_scales_bound_units"] == 1


def test_dense_e4m3_selection_needs_no_scales(tmp_path):
    assignment = _assignment(
        tmp_path, formats={DENSE_E4M3: "TESSERA_E4M3_K1_R1024"},
        hessian_block="modern-weights-only")
    report = export.require_priced_export_inputs(assignment)
    assert report["input_scales_required"] is False
    assert report["w4a4_units"] == 0


# -- #204: the VALUE the file carries must be the value that priced the row.

@pytest.mark.parametrize("served", [1.0, 10000.0])
def test_a_scale_that_differs_from_the_priced_scale_refuses(tmp_path, served):
    """codex ``prismaquant_seam_inputs.py`` case 3: G = 1 and G = 10000 at the
    same key.  The pre-#204 gate checked that the key existed; the served
    activation quantisation differs by four orders of magnitude between the
    two files, at a unit whose cost row was scored under one specific scale.
    """
    assignment = _assignment(
        tmp_path, formats={DENSE_E2M1: "TESSERA_E2M1_K2_R896"},
        hessian_block="modern-weights-only")
    scales = _scales_file(tmp_path, [DENSE_E2M1], value=served)
    with pytest.raises(export.TesseraExportLaneError,
                       match=r"input_global_scale = .* but the allocation "
                             r"priced .*" + DENSE_E2M1.replace(".", r"\.")):
        export.require_priced_export_inputs(
            assignment, input_scales_path=scales)


def test_the_priced_scale_is_accepted_and_counted(tmp_path):
    assignment = _assignment(
        tmp_path, formats={DENSE_E2M1: "TESSERA_E2M1_K2_R896",
                           "model.layers.1.mlp.up_proj": "TESSERA_E2M1_K2_R896"},
        hessian_block="modern-weights-only")
    scales = _scales_file(tmp_path, [DENSE_E2M1, "model.layers.1.mlp.up_proj"])
    report = export.require_priced_export_inputs(
        assignment, input_scales_path=scales)
    assert report["input_scales_bound_units"] == 2


def test_an_allocation_without_priced_scales_is_unbound(tmp_path):
    """A pre-#204 allocation carries no ``tessera_activation_static_scales``
    block; the file's value has nothing to be compared against, and that is
    refused by name rather than accepted on key presence."""
    assignment = _assignment(
        tmp_path, formats={DENSE_E2M1: "TESSERA_E2M1_K2_R896"},
        hessian_block="modern-weights-only", scales=None)
    scales = _scales_file(tmp_path, [DENSE_E2M1])
    with pytest.raises(export.TesseraExportLaneError,
                       match="no tessera_activation_static_scales.*unbound"):
        export.require_priced_export_inputs(
            assignment, input_scales_path=scales)


def test_a_w4a4_unit_the_allocation_priced_no_scale_for_is_unbound(tmp_path):
    """The block exists but names another unit: the selected unit's row
    carried no ``input_global_scale`` (or the allocator lost it)."""
    assignment = _assignment(
        tmp_path, formats={DENSE_E2M1: "TESSERA_E2M1_K2_R896"},
        hessian_block="modern-weights-only",
        scales={"model.layers.1.mlp.up_proj": SCALE})
    scales = _scales_file(tmp_path, [DENSE_E2M1])
    with pytest.raises(export.TesseraExportLaneError,
                       match="priced no input_global_scale for.*"
                             + DENSE_E2M1.replace(".", r"\.")):
        export.require_priced_export_inputs(
            assignment, input_scales_path=scales)


def test_a_non_scalar_scale_tensor_refuses(tmp_path):
    from safetensors.torch import save_file

    assignment = _assignment(
        tmp_path, formats={DENSE_E2M1: "TESSERA_E2M1_K2_R896"},
        hessian_block="modern-weights-only")
    scales = tmp_path / "input_scales.safetensors"
    save_file({f"{DENSE_E2M1}.input_global_scale":
               torch.full((2,), SCALE, dtype=torch.float32)}, str(scales))
    with pytest.raises(export.TesseraExportLaneError,
                       match="one scalar per unit"):
        export.require_priced_export_inputs(
            assignment, input_scales_path=scales)


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
