"""Behaviour tests for the trellis AURA dW store.

Every test drives real repo code:
``aura_cost._stored_production_anchor_delta`` for the subtraction contract,
``format_registry.canonical_format_name`` for the plan comparison
``compute_aura_cost_streamed`` performs, and
``trellis_allocator.build_trellis_allocator_candidate`` for wire pricing.
No test greps a source file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from prismaquant import aura_cost
from prismaquant import format_registry as fr
from prismaquant.trellis_anchor_dw import (
    _source_identity,
    TRELLIS_ANCHOR_DW_ENV,
    TRELLIS_ANCHOR_DW_STORE_SCHEMA,
    TrellisAnchorDeltaSource,
    TrellisAnchorStoreError,
    open_trellis_anchor_source,
    write_trellis_anchor_store,
)
from prismaquant.trellis_formats import (
    E2M1_FAMILY,
    LAYOUT_FIXED_QUOTA,
    native_code_value,
)

QNAME = "model.layers.0.self_attn.o_proj"
ROWS, COLUMNS = 8, 256
RATE_Q256 = 2 * 256  # a whole-bit rung: R2 on a 256-column tensor


def _alphabet(rate: int) -> list[int]:
    codes = list(range(1 << (rate + 1)))
    return sorted(
        codes, key=lambda code: (native_code_value(E2M1_FAMILY, code), code),
    )


def _encoder_identity() -> dict[str, str]:
    return {
        "encoder_snapshot_tree_sha256": "a" * 64,
        "container_image": "example/spark-vllm@sha256:" + "b" * 64,
        "torch_version": "2.13.0+cu130",
        "device_name": "NVIDIA GB10",
    }


def _weights(seed: int = 20260829) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(
        ROWS, COLUMNS, generator=generator, dtype=torch.float32,
    )
    # Stand-in for an encoder output. The store is agnostic to how the
    # rendered tensor was produced; that is exactly the separation under test.
    rendered = source + 0.01 * torch.randn(
        ROWS, COLUMNS, generator=generator, dtype=torch.float32,
    )
    return source, rendered


def _entry(
    source: torch.Tensor,
    rendered: torch.Tensor,
    *,
    qname: str = QNAME,
    format_name: str = f"TCQ_E2M1_R{RATE_Q256}",
    rate_q256: int = RATE_Q256,
    family: str = E2M1_FAMILY,
) -> dict[str, object]:
    shaped_rate = rate_q256 // 256
    return {
        "qname": qname,
        "format": format_name,
        "family": family,
        "body_rate_q256": rate_q256,
        "layout": LAYOUT_FIXED_QUOTA,
        "alphabets": {shaped_rate: _alphabet(shaped_rate)},
        "rendered_weight": rendered,
        "source_weight": source,
    }


def _store(tmp_path: Path, **overrides) -> Path:
    source, rendered = _weights()
    entry = _entry(source, rendered)
    entry.update(overrides)
    write_trellis_anchor_store(
        tmp_path,
        encoder_identity=_encoder_identity(),
        entries=[entry],
    )
    return tmp_path


def _module(weight: torch.Tensor) -> nn.Linear:
    linear = nn.Linear(COLUMNS, ROWS, bias=False)
    with torch.no_grad():
        linear.weight.copy_(weight)
    return linear


# --------------------------------------------------------------------------
# The thing the whole design rests on: a stored render becomes an AURA number.
# --------------------------------------------------------------------------

def test_replayed_render_yields_the_hand_computed_aura_predicted_dloss(
    tmp_path,
):
    """store -> render_layer -> real dW subtraction -> 0.5*mean_k <g,dW>^2.

    The arithmetic is aura_cost's own (``_stored_production_anchor_delta`` for
    the fp32-subtract-then-store contract; the harvest formula from its module
    docstring), so this asserts the store composes with the real consumer, not
    with a restatement of it.
    """
    source, rendered = _weights()
    root = _store(tmp_path)
    provider = TrellisAnchorDeltaSource(root)
    fmt = f"TCQ_E2M1_R{RATE_Q256}"

    replayed = provider.render_layer(
        layer=0,
        modules={QNAME: _module(source)},
        formats_by_qname={QNAME: [fmt]},
    )
    assert set(replayed) == {(QNAME, fmt)}
    assert torch.equal(replayed[(QNAME, fmt)], rendered)

    delta = aura_cost._stored_production_anchor_delta(
        replayed[(QNAME, fmt)],
        source,
        storage_dtype=torch.float32,
    )
    assert torch.allclose(delta, rendered - source, atol=0.0, rtol=0.0)

    generator = torch.Generator().manual_seed(7)
    grads = [
        torch.randn(ROWS, COLUMNS, generator=generator, dtype=torch.float32)
        for _ in range(3)
    ]
    samples = [float((g * delta).sum().item()) ** 2 for g in grads]
    predicted_dloss = 0.5 * (sum(samples) / len(samples))

    # Hand-computed the other way round: the inner product is linear in dW, so
    # scaling the render's error by c scales predicted_dloss by c**2. A cost
    # that failed to track that would not be a second-order KL contribution.
    scaled_delta = 2.0 * delta
    scaled = 0.5 * (
        sum(float((g * scaled_delta).sum().item()) ** 2 for g in grads)
        / len(grads)
    )
    assert predicted_dloss > 0.0
    assert scaled == pytest.approx(4.0 * predicted_dloss, rel=1e-9)


def test_a_zero_error_render_prices_at_zero(tmp_path):
    """A lossless render must cost nothing, whatever the gradients are."""
    source, _ = _weights()
    _store(tmp_path, rendered_weight=source.clone())
    provider = TrellisAnchorDeltaSource(tmp_path)
    replayed = provider.render_layer(
        layer=0,
        modules={QNAME: _module(source)},
        formats_by_qname={QNAME: [f"TCQ_E2M1_R{RATE_Q256}"]},
    )
    delta = aura_cost._stored_production_anchor_delta(
        replayed[(QNAME, f"TCQ_E2M1_R{RATE_Q256}")],
        source,
        storage_dtype=torch.float32,
    )
    g = torch.randn(ROWS, COLUMNS, generator=torch.Generator().manual_seed(3))
    assert float((g * delta).sum().item()) == 0.0


# --------------------------------------------------------------------------
# The confound guard: a dW is only meaningful against the weights it was
# encoded from.
# --------------------------------------------------------------------------

def test_a_live_weight_that_is_not_the_encode_source_refuses(tmp_path):
    source, _ = _weights()
    _store(tmp_path)
    provider = TrellisAnchorDeltaSource(tmp_path)
    drifted = source.clone()
    drifted[0, 0] += 1e-3
    with pytest.raises(TrellisAnchorStoreError, match="encoded against"):
        provider.render_layer(
            layer=0,
            modules={QNAME: _module(drifted)},
            formats_by_qname={QNAME: [f"TCQ_E2M1_R{RATE_Q256}"]},
        )


def test_a_live_weight_of_a_different_shape_refuses(tmp_path):
    _store(tmp_path)
    provider = TrellisAnchorDeltaSource(tmp_path)
    other = nn.Linear(COLUMNS, ROWS + 1, bias=False)
    with pytest.raises(TrellisAnchorStoreError, match="shape"):
        provider.render_layer(
            layer=0,
            modules={QNAME: other},
            formats_by_qname={QNAME: [f"TCQ_E2M1_R{RATE_Q256}"]},
        )


def test_the_recorded_source_identity_is_the_one_aura_would_compute(tmp_path):
    """The store's binding must agree with AURA's own fallback hash.

    aura_cost falls back to ``_source_weight_value_identity`` for injected
    renderers; if the store hashed differently, every replay would look like
    drift or, worse, drift would look like a match.
    """
    from prismaquant.production_weight_cache import (
        _source_weight_value_identity,
    )

    source, _ = _weights()
    _store(tmp_path)
    provider = TrellisAnchorDeltaSource(tmp_path)
    shape, sha = _source_weight_value_identity(source)
    recorded = provider.source_weight_identity_for(QNAME)
    assert recorded == {"shape": shape, "sha256": sha}


# --------------------------------------------------------------------------
# Plan and content integrity.
# --------------------------------------------------------------------------

def test_an_unplanned_qname_refuses(tmp_path):
    source, _ = _weights()
    _store(tmp_path)
    provider = TrellisAnchorDeltaSource(tmp_path)
    with pytest.raises(TrellisAnchorStoreError, match="unplanned"):
        provider.render_layer(
            layer=0,
            modules={"model.layers.9.mlp.down_proj": _module(source)},
            formats_by_qname={
                "model.layers.9.mlp.down_proj": [f"TCQ_E2M1_R{RATE_Q256}"],
            },
        )


def test_an_unstored_rung_refuses(tmp_path):
    source, _ = _weights()
    _store(tmp_path)
    provider = TrellisAnchorDeltaSource(tmp_path)
    with pytest.raises(TrellisAnchorStoreError, match="asked for"):
        provider.render_layer(
            layer=0,
            modules={QNAME: _module(source)},
            formats_by_qname={QNAME: ["TCQ_E2M1_R768"]},
        )


def test_a_tampered_shard_refuses(tmp_path):
    source, rendered = _weights()
    _store(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    shard = tmp_path / manifest["records"][0]["shard"]
    poisoned = rendered.clone()
    poisoned[0, 0] += 1.0
    torch.save(poisoned, shard)
    provider = TrellisAnchorDeltaSource(tmp_path)
    with pytest.raises(TrellisAnchorStoreError, match="shard content"):
        provider.render_layer(
            layer=0,
            modules={QNAME: _module(source)},
            formats_by_qname={QNAME: [f"TCQ_E2M1_R{RATE_Q256}"]},
        )


def test_a_reshaped_shard_with_identical_bytes_refuses(tmp_path):
    """A raw byte hash cannot tell [8,256] from [256,8]; the digest must."""
    source, rendered = _weights()
    _store(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    shard = tmp_path / manifest["records"][0]["shard"]
    torch.save(rendered.t().contiguous(), shard)
    provider = TrellisAnchorDeltaSource(tmp_path)
    with pytest.raises(TrellisAnchorStoreError, match="shard shape"):
        provider.render_layer(
            layer=0,
            modules={QNAME: _module(source)},
            formats_by_qname={QNAME: [f"TCQ_E2M1_R{RATE_Q256}"]},
        )


def test_a_resigned_manifest_with_wrong_bytes_refuses(tmp_path):
    """Re-signing hides the edit from the identity check, so the byte count
    must be re-derived on open rather than trusted."""
    _store(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["records"][0]["payload_bytes"] += 1
    body = {
        key: manifest[key]
        for key in ("schema", "encoder_identity", "records")
    }
    manifest["identity_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(TrellisAnchorStoreError, match="the allocator prices"):
        TrellisAnchorDeltaSource(tmp_path)


def test_an_edited_manifest_refuses(tmp_path):
    _store(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["records"][0]["payload_bytes"] += 1
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(TrellisAnchorStoreError, match="identity_sha256"):
        TrellisAnchorDeltaSource(tmp_path)


def test_render_count_tracks_replayed_pairs(tmp_path):
    source, _ = _weights()
    _store(tmp_path)
    provider = TrellisAnchorDeltaSource(tmp_path)
    assert provider.render_count == 0
    for expected in (1, 2):
        provider.render_layer(
            layer=0,
            modules={QNAME: _module(source)},
            formats_by_qname={QNAME: [f"TCQ_E2M1_R{RATE_Q256}"]},
        )
        assert provider.render_count == expected
    assert provider.max_live_rendered == 1


# --------------------------------------------------------------------------
# Wire expressibility: the store cannot hold a rung wire.v1 cannot carry.
# --------------------------------------------------------------------------

def test_a_ragged_column_count_refuses(tmp_path):
    """The store refuses a shape the wire helper will not schedule.

    The rule belongs to ``uniform_column_schedule``; what is asserted here is
    that the store surfaces it as a refusal to write rather than storing a
    render the allocator could never price.
    """
    generator = torch.Generator().manual_seed(11)
    source = torch.randn(4, 300, generator=generator)
    with pytest.raises(
        TrellisAnchorStoreError, match="not expressible as a",
    ) as excinfo:
        write_trellis_anchor_store(
            tmp_path,
            encoder_identity=_encoder_identity(),
            entries=[_entry(source, source.clone())],
        )
    assert "multiple of 256" in str(excinfo.value)
    assert not (tmp_path / "manifest.json").exists()


def test_a_non_trellis_format_refuses(tmp_path):
    source, rendered = _weights()
    with pytest.raises(TrellisAnchorStoreError, match="not a trellis rung"):
        write_trellis_anchor_store(
            tmp_path,
            encoder_identity=_encoder_identity(),
            entries=[_entry(source, rendered, format_name="NVFP4")],
        )


def test_a_format_name_that_contradicts_its_rate_refuses(tmp_path):
    source, rendered = _weights()
    with pytest.raises(TrellisAnchorStoreError, match="spells"):
        write_trellis_anchor_store(
            tmp_path,
            encoder_identity=_encoder_identity(),
            entries=[_entry(source, rendered, format_name="TCQ_E2M1_R768")],
        )


def test_stored_payload_bytes_equal_what_the_allocator_charges(tmp_path):
    """The store must not become a second byte-accounting authority."""
    from prismaquant.trellis_allocator import build_trellis_allocator_candidate
    from prismaquant.trellis_rate_surface import uniform_column_schedule

    _store(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    record = manifest["records"][0]
    candidate = build_trellis_allocator_candidate(
        QNAME,
        (ROWS, COLUMNS),
        family=E2M1_FAMILY,
        body_rate_q256=RATE_Q256,
        layout=LAYOUT_FIXED_QUOTA,
        schedule=uniform_column_schedule(COLUMNS, RATE_Q256, family=E2M1_FAMILY),
        alphabets={RATE_Q256 // 256: tuple(_alphabet(RATE_Q256 // 256))},
        predicted_dloss=0.0,
        qname=QNAME,
    )
    assert record["payload_bytes"] == int(
        candidate.footprint["total_bytes"]
    )


# --------------------------------------------------------------------------
# The encode environment travels with the bytes.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dropped", [
    "encoder_snapshot_tree_sha256",
    "container_image",
    "torch_version",
    "device_name",
])
def test_an_incomplete_encoder_identity_refuses(tmp_path, dropped):
    source, rendered = _weights()
    identity = _encoder_identity()
    identity.pop(dropped)
    with pytest.raises(TrellisAnchorStoreError, match="encoder identity"):
        write_trellis_anchor_store(
            tmp_path,
            encoder_identity=identity,
            entries=[_entry(source, rendered)],
        )


def test_the_store_declares_no_cost_currency_and_no_runtime_contract(tmp_path):
    """One DP prices in one currency, and it is not this module's to name."""
    _store(tmp_path)
    identity = TrellisAnchorDeltaSource(tmp_path).identity
    assert identity["schema"] == TRELLIS_ANCHOR_DW_STORE_SCHEMA
    assert identity["declares_cost_currency"] is False
    assert identity["declares_runtime_contract"] is False
    assert identity["encoder_identity"] == _encoder_identity()
    flat = json.dumps(identity)
    for forbidden in ("activation_contract", "route_status", "backed"):
        assert forbidden not in flat


# --------------------------------------------------------------------------
# The plan surface aura_cost actually compares against.
# --------------------------------------------------------------------------

def test_trellis_rung_names_survive_registry_canonicalization(tmp_path):
    """compute_aura_cost_streamed compares the renderer's plan to the AURA
    plan after ``fr.canonical_format_name``. If a future registry change
    aliased a TCQ name, that comparison would fail and this catches it."""
    _store(tmp_path)
    provider = TrellisAnchorDeltaSource(tmp_path)
    plan = provider.formats_by_qname
    assert plan == {QNAME: (f"TCQ_E2M1_R{RATE_Q256}",)}
    canonicalized = {
        name: tuple(fr.canonical_format_name(f) for f in formats)
        for name, formats in plan.items()
    }
    assert canonicalized == plan


def test_a_trellis_rung_is_still_absent_from_the_format_registry():
    """The store supplies dW without a FormatSpec, which is the point; if a
    TCQ name ever became a registered format, ``UNWIRED_LINKS`` entry 1 and
    this module's reason for existing both change."""
    with pytest.raises(KeyError):
        fr.get_format(f"TCQ_E2M1_R{RATE_Q256}")


# --------------------------------------------------------------------------
# Default path.
# --------------------------------------------------------------------------

def test_unset_flag_yields_no_source(monkeypatch):
    monkeypatch.delenv(TRELLIS_ANCHOR_DW_ENV, raising=False)
    assert open_trellis_anchor_source() is None


def test_the_flag_opens_the_store(tmp_path, monkeypatch):
    _store(tmp_path)
    monkeypatch.setenv(TRELLIS_ANCHOR_DW_ENV, str(tmp_path))
    provider = open_trellis_anchor_source()
    assert isinstance(provider, TrellisAnchorDeltaSource)
    assert set(provider.formats_by_qname) == {QNAME}


def test_importing_this_module_registers_nothing_global():
    """No format, no render mechanism, no serving lane. A research module that
    mutated a global registry on import would not be opt-in."""
    from prismaquant import render_score

    before_formats = {spec.name for spec in fr.list_formats()}
    before_mechanisms = set(render_score.registered_render_mechanisms())

    # Execute a FRESH copy under a throwaway name rather than reloading the
    # live module.  ``importlib.reload`` re-executes into the SAME module
    # __dict__, so every already-imported function would start resolving a
    # brand-new ``TrellisAnchorStoreError`` while this test module still holds
    # the old class -- an order-dependent failure in whatever runs next.
    import importlib.util

    module = importlib.import_module("prismaquant.trellis_anchor_dw")
    spec = importlib.util.spec_from_file_location(
        "prismaquant._trellis_anchor_dw_import_probe", module.__file__,
    )
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module out of sys.modules, so the
    # throwaway name has to be registered while the copy executes -- and
    # removed straight after, so nothing else can pick it up.
    sys.modules[spec.name] = probe
    try:
        spec.loader.exec_module(probe)
    finally:
        sys.modules.pop(spec.name, None)

    assert {spec.name for spec in fr.list_formats()} == before_formats
    assert set(render_score.registered_render_mechanisms()) == (
        before_mechanisms
    )
    # And the live module is untouched by the probe.
    assert sys.modules["prismaquant.trellis_anchor_dw"] is module
    assert module.TrellisAnchorStoreError is TrellisAnchorStoreError


# --------------------------------------------------------------------------
# The multi-rung store is the NORMAL case: trellis_menu refuses a unit with
# fewer than two anchors, so the single-record store every other test writes
# is the degenerate one.  These cover the shape the design actually needs.
# --------------------------------------------------------------------------

RATE_Q256_HIGH = 3 * 256
FMT_LOW = f"TCQ_E2M1_R{RATE_Q256}"
FMT_HIGH = f"TCQ_E2M1_R{RATE_Q256_HIGH}"


def _two_rung_store(tmp_path: Path, *, high_source: torch.Tensor | None = None):
    """Two rungs of one qname -- the production shape.

    ``high_source`` lets a test encode the second rung against a DIFFERENT
    weight, which is the incoherent store the coherence guard exists for.
    """

    source, rendered_low = _weights()
    rendered_high = source + 0.001 * (rendered_low - source)
    entries = [
        _entry(source, rendered_low),
        _entry(
            source if high_source is None else high_source,
            rendered_high,
            format_name=FMT_HIGH,
            rate_q256=RATE_Q256_HIGH,
        ),
    ]
    write_trellis_anchor_store(
        tmp_path, encoder_identity=_encoder_identity(), entries=entries,
    )
    return source, rendered_low, rendered_high


def test_two_rungs_of_one_unit_replay_together(tmp_path):
    source, rendered_low, rendered_high = _two_rung_store(tmp_path)
    provider = TrellisAnchorDeltaSource(tmp_path)
    assert provider.formats_by_qname == {QNAME: (FMT_LOW, FMT_HIGH)}

    replayed = provider.render_layer(
        layer=0,
        modules={QNAME: _module(source)},
        formats_by_qname={QNAME: [FMT_LOW, FMT_HIGH]},
    )
    assert set(replayed) == {(QNAME, FMT_LOW), (QNAME, FMT_HIGH)}
    assert provider.render_count == 2
    assert provider.max_live_rendered == 2

    # Each rung replays ITS OWN render, not whichever the dict found first.
    assert torch.equal(replayed[(QNAME, FMT_LOW)].float(), rendered_low)
    assert torch.equal(replayed[(QNAME, FMT_HIGH)].float(), rendered_high)

    # And AURA prices them apart: the higher rung is the smaller perturbation,
    # so its predicted_dloss is strictly lower.  This is the whole point of
    # holding two anchors -- one point cannot define a rate surface.
    grad = torch.randn(
        ROWS, COLUMNS, generator=torch.Generator().manual_seed(7),
    )
    priced = {}
    for fmt in (FMT_LOW, FMT_HIGH):
        delta = aura_cost._stored_production_anchor_delta(
            replayed[(QNAME, fmt)], source, storage_dtype=torch.float32,
        )
        priced[fmt] = 0.5 * float((grad * delta).sum().item()) ** 2
    assert priced[FMT_HIGH] < priced[FMT_LOW]


def test_two_rungs_encoded_against_different_sources_refuse_at_write(tmp_path):
    other, _ = _weights(seed=20260830)
    with pytest.raises(
        TrellisAnchorStoreError, match="must share one source weight",
    ):
        _two_rung_store(tmp_path, high_source=other)
    assert not (tmp_path / "manifest.json").exists()


def test_two_rungs_encoded_against_different_sources_refuse_at_open(tmp_path):
    """A store this repo did not write must be refused on open too.

    The offline encode driver runs in another container; the open-side guard
    is the only one that sees what it actually produced.
    """

    _two_rung_store(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    other, _ = _weights(seed=20260830)
    other_shape, other_sha = _source_identity(other)
    for record in manifest["records"]:
        if record["format"] == FMT_HIGH:
            record["source_shape"] = list(other_shape)
            record["source_sha256"] = other_sha
    body = {
        key: manifest[key]
        for key in ("schema", "encoder_identity", "records")
    }
    manifest["identity_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    with pytest.raises(
        TrellisAnchorStoreError, match="must share one source weight",
    ):
        TrellisAnchorDeltaSource(tmp_path)
