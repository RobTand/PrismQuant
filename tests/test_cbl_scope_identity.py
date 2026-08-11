"""CPU-only regressions for mixed lattice/learned CB producer identity."""

from __future__ import annotations

import copy
import hashlib
import json
import platform

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant import nvfp4_cb_formats as cb
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_fields_for_context,
    cb_serialization_context_from_env,
    cb_serialization_context_from_stamp,
    cb_serialization_context_stamp,
    cb_tensor_payload_breakdown,
    codebook_source_for_format,
    lattice_codebook_content_sha256,
    scale_sweep_for_format,
    validate_cb_serialization_context_stamp,
)


_NV = "NVFP4_CB_K12"
_FP8 = "FP8_CBL_K28"
_FP8_LATTICE = "FP8_CB_K28"
_QNAME = "model.layers.0.mlp.gate_proj"


def _canonical_sha256(value: object) -> tuple[int, str]:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _mixed_refs_and_digests():
    nv_refs = tuple(f"cb_codebook.lattice.{_NV}.sub{i}" for i in range(2))
    fp8_refs = tuple(
        f"cb_codebook.layer0.gate_proj.{_FP8}.sub{i}" for i in range(4)
    )
    refs = {_QNAME: {_NV: nv_refs, _FP8: fp8_refs}}
    digests = {
        **dict(zip(nv_refs, lattice_codebook_content_sha256(_NV), strict=True)),
        **{ref: f"{index + 1:064x}" for index, ref in enumerate(fp8_refs)},
    }
    return refs, digests


def _mixed_context(*, scale_sweep_scope: str = "all"):
    refs, digests = _mixed_refs_and_digests()
    return CBSerializationContext.production(
        codebook_source_scope="fp8",
        scale_sweep_scope=scale_sweep_scope,
        codebook_refs_by_qname_format=refs,
        codebook_content_digests=digests,
    )


def test_unset_scopes_pin_76666bd_stamp_and_rendered_bytes():
    """New optional identity must not perturb the v0.10.0 lattice producer."""
    context = cb_serialization_context_from_env({})
    assert context.codebook_source_scope is None
    assert context.scale_sweep_scope is None

    stamp = cb_serialization_context_stamp(
        context, formats=[_NV, _FP8_LATTICE]
    )
    assert "codebook_source_scope" not in stamp
    assert "scale_sweep_scope" not in stamp
    assert _canonical_sha256(stamp) == (
        833,
        "0077c86757b4b8ab1ca3a24642baa5336317c0422dfac12454801f968b8edfe7",
    )

    # These hashes pin every packed tensor byte from 76666bd.  A direct
    # equality against a second call through the new helper would only prove
    # self-consistency, not backward identity.
    #
    # The FP8 row scale is pinned by VALUE, not by digest.  It is the scalar
    # argmin of a scale sweep whose objective is a reduction over all 256
    # columns, and that reduction is not bit-reproducible across CPU
    # architectures: the recorded digest below reproduces exactly on the
    # aarch64 build box and differs in the low bits on x86 CI, while the packed
    # indices -- the bytes that actually ship -- are identical on both.  The
    # bound is float32's own worst case for reordering an n-term sum,
    # n * 2**-23, not a tuned constant.  The exact digest is still asserted
    # where it was recorded, so the strongest claim keeps running on the
    # machine that builds artifacts.
    weight = torch.linspace(-0.5, 0.5, 256, dtype=torch.float32).reshape(1, 256)
    col_weights = torch.linspace(0.1, 1.0, 256, dtype=torch.float32)
    reduction_rtol = weight.shape[1] * 2 ** -23
    expected_packed = {
        _NV: "5434e7fb94b22160209b2692b94a6285af417bbfe61ea3a3f78207cb73678bde",
        _FP8_LATTICE: "5d8dba3c2a76e3d46e564b2aa63777a1be53cc431d3772e5cbac14ead2b41ba9",
    }
    expected_fp8_scale = 0.0021216266322880983
    expected_fp8_scale_digest_on_aarch64 = (
        "620ea8dae04d1794e36f7322520386a450d05b41d03ff0f7ae573db0ecd33d59")
    for format_name in (_NV, _FP8_LATTICE):
        grid = "fp4" if format_name == _NV else "fp8"
        k = int(format_name.rsplit("K", 1)[1])
        fields = cb_fields_for_context(
            fr.get_format(format_name),
            weight,
            context=context,
            col_weights=col_weights,
        )
        packed = cb.nvfp4_cb_assemble_bytes(fields, k, grid, "product")
        assert hashlib.sha256(
            _tensor_bytes(packed)).hexdigest() == expected_packed[format_name]
        if grid != "fp8":
            continue
        scales = fields["scales"].to(torch.float32)
        torch.testing.assert_close(
            scales,
            torch.full_like(scales, expected_fp8_scale),
            rtol=reduction_rtol,
            atol=0.0,
        )
        if platform.machine() == "aarch64":
            assert hashlib.sha256(
                _tensor_bytes(scales)
            ).hexdigest() == expected_fp8_scale_digest_on_aarch64


def test_homogeneous_explicit_scopes_canonicalize_to_old_stamp():
    baseline = cb_serialization_context_stamp(
        CBSerializationContext.production(), formats=[_NV, _FP8_LATTICE]
    )
    explicit = cb_serialization_context_stamp(
        CBSerializationContext.production(
            codebook_source_scope="none",
            scale_sweep_scope="all",
        ),
        formats=[_NV, _FP8_LATTICE],
    )
    assert explicit == baseline


def test_env_scopes_are_authoritative_and_all_warns():
    with pytest.raises(ValueError, match="requires CB_CODEBOOK_BUNDLE"):
        cb_serialization_context_from_env({
            "CB_CODEBOOK_SOURCE_SCOPE": "fp8",
            "CB_SCALE_SWEEP_SCOPE": "nvfp4",
        })
    context = _mixed_context(scale_sweep_scope="nvfp4")
    assert context.codebook_source == "learned"
    assert context.scale_sweep is True
    assert codebook_source_for_format(_NV, context) == "lattice"
    assert codebook_source_for_format(_FP8, context) == "learned"
    assert scale_sweep_for_format(_NV, context) is True
    assert scale_sweep_for_format(_FP8, context) is False

    with pytest.warns(RuntimeWarning, match="NVFP4.*NO-GO"):
        all_context = CBSerializationContext.production(
            codebook_source_scope="all",
        )
    assert codebook_source_for_format(_NV, all_context) == "lattice"

    with pytest.raises(ValueError, match="inconsistent"):
        cb_serialization_context_from_env({
            "CB_CODEBOOK_SOURCE": "lattice",
            "CB_CODEBOOK_SOURCE_SCOPE": "fp8",
        })
    with pytest.raises(ValueError, match="two_tier requires scale_sweep=True"):
        cb_serialization_context_from_env({"CB_SCALE_SWEEP_SCOPE": "fp8"})


def test_mixed_stamp_roundtrip_binds_family_scopes_refs_and_complete_digests():
    context = _mixed_context(scale_sweep_scope="nvfp4")
    refs, digests = _mixed_refs_and_digests()
    stamp = cb_serialization_context_stamp(context, formats=[_NV, _FP8])

    assert stamp["codebook_source"] == "learned"
    assert stamp["codebook_source_scope"] == "fp8"
    assert stamp["scale_sweep"] is True
    assert stamp["scale_sweep_scope"] == "nvfp4"
    assert stamp["lattice_codebook_sha256_by_format"] == {
        _NV: list(lattice_codebook_content_sha256(_NV))
    }
    assert stamp["codebook_content_sha256"] == dict(sorted(digests.items()))
    assert stamp["codebook_refs_by_qname_format"] == {
        _QNAME: {name: list(value) for name, value in sorted(refs[_QNAME].items())}
    }

    restored = cb_serialization_context_from_stamp(stamp, where="scope test")
    validate_cb_serialization_context_stamp(stamp, restored, where="scope test")
    assert restored.codebook_refs_by_qname_format == refs

    nv = cb_tensor_payload_breakdown(
        _NV, (1, 256), qname=_QNAME, context=restored
    )
    fp8 = cb_tensor_payload_breakdown(
        _FP8, (1, 256), qname=_QNAME, context=restored
    )
    assert nv["sidecar_identity"]["codebook_source"] == "lattice"
    assert fp8["sidecar_identity"]["codebook_source"] == "learned"
    assert tuple(nv["sidecar_identity"]["codebook_ref"]) == refs[_QNAME][_NV]
    assert tuple(fp8["sidecar_identity"]["codebook_ref"]) == refs[_QNAME][_FP8]
    assert nv["identity"]["scale_sweep"] is True
    assert fp8["identity"]["scale_sweep"] is False


def test_mixed_identity_fails_closed_on_missing_values_refs_or_scope_drift():
    with pytest.raises(ValueError, match="codebook_content_digests"):
        cb_serialization_context_stamp(
            CBSerializationContext.production(codebook_source_scope="fp8")
        )

    context = _mixed_context(scale_sweep_scope="nvfp4")
    stamp = cb_serialization_context_stamp(context, formats=[_NV, _FP8])

    missing_digest = copy.deepcopy(stamp)
    missing_digest["codebook_content_sha256"].pop(next(iter(
        context.codebook_content_digests
    )))
    with pytest.raises(ValueError, match="missing="):
        validate_cb_serialization_context_stamp(
            missing_digest, context, where="missing digest"
        )

    missing_refs = copy.deepcopy(stamp)
    missing_refs.pop("codebook_refs_by_qname_format")
    with pytest.raises(ValueError, match="per-qname/format codebook refs"):
        validate_cb_serialization_context_stamp(
            missing_refs, context, where="missing refs"
        )

    with pytest.warns(RuntimeWarning, match="NVFP4.*NO-GO"):
        wrong_source_scope = CBSerializationContext.production(
            codebook_source_scope="all",
            scale_sweep_scope="nvfp4",
            codebook_refs_by_qname_format=context.codebook_refs_by_qname_format,
            codebook_content_digests=context.codebook_content_digests,
        )
    with pytest.raises(ValueError, match="codebook source scope differs"):
        validate_cb_serialization_context_stamp(
            stamp, wrong_source_scope, where="source scope drift"
        )

    wrong_sweep_scope = _mixed_context(scale_sweep_scope="all")
    with pytest.raises(ValueError, match="scale-sweep scope differs"):
        validate_cb_serialization_context_stamp(
            stamp, wrong_sweep_scope, where="sweep scope drift"
        )
