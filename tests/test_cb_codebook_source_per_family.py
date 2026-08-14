"""CPU-only regression tests for per-family CB codebook-source plumbing."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant.cb_export_config import build_quant_config, codebook_tensors
from prismaquant.cb_warm_state import (
    encoder_initializer_identity,
    warm_serialization_context,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    cb_fields_for_context,
    cb_serialization_context_from_stamp,
    cb_serialization_context_stamp,
    parse_codebook_source_setting,
    validate_cb_cost_provenance,
    validate_cb_serialization_context_stamp,
)
from prismaquant.production_weight_cache import (
    bind_cb_render_identity_source_weights,
    build_production_cache_cb_render_identity,
    validate_cb_render_identity_metadata,
)


FP4_FORMAT = "NVFP4_CB_K12"
FP8_FORMAT = "FP8_CB_K28"
FP4_QNAME = "layer.q_proj"
FP8_QNAME = "layer.v_proj"


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()


def _mixed_context() -> CBSerializationContext:
    fp8_refs = tuple(
        f"cb_codebook.v_proj.{FP8_FORMAT}.sub{index}"
        for index in range(4)
    )
    fp8_digest = hashlib.sha256(
        torch.zeros(128, 2, dtype=torch.float16).numpy().tobytes()
    ).hexdigest()
    return CBSerializationContext.production(
        codebook_source_by_family={
            "fp4": "lattice",
            "fp8": "learned",
        },
        codebook_refs={FP8_QNAME: fp8_refs},
        codebook_content_digests={
            ref: fp8_digest for ref in fp8_refs
        },
    )


def _books(fp8_ref: str):
    codebooks = {
        ("lattice", FP4_FORMAT): (
            torch.zeros(64, 4),
            torch.zeros(64, 4),
        ),
        (fp8_ref, FP8_FORMAT): tuple(
            torch.zeros(128, 2) for _ in range(4)
        ),
    }
    blobs = {
        name: tensor
        for (ref, fmt), codebook in codebooks.items()
        for name, tensor in codebook_tensors(ref, fmt, codebook).items()
    }
    return codebooks, blobs


def _config(
    context: CBSerializationContext,
    *,
    fp8_ref: str,
) -> dict:
    codebooks, blobs = _books(fp8_ref)
    return build_quant_config(
        assignment={FP4_QNAME: FP4_FORMAT, FP8_QNAME: FP8_FORMAT},
        cb_targets={
            FP4_QNAME: ("fp4", "product", 12),
            FP8_QNAME: ("fp8", "product", 28),
        },
        source_targets=[],
        stock_targets={},
        by_group={
            ("lattice", FP4_FORMAT): [FP4_QNAME],
            (fp8_ref, FP8_FORMAT): [FP8_QNAME],
        },
        codebooks=codebooks,
        col_weights={},
        codebook_tensors_by_name=blobs,
        ignore=[],
        codebook_file="cb.pqcb",
        scale_coding="two_tier",
        codebook_source=context.codebook_source,
        serialized_payload_summary={"total_bytes": 123},
        serialization_context=context,
        cb_render_identity=None,
        git_commit="test",
    )


def test_scalar_setting_and_outputs_remain_byte_identical():
    assert parse_codebook_source_setting(
        "lattice", where="test"
    ) == ("lattice", None)
    context = CBSerializationContext.production()
    assert context.codebook_source_by_family is None

    stamp = cb_serialization_context_stamp(
        context, formats=[FP4_FORMAT, FP8_FORMAT]
    )
    assert "codebook_source_by_family" not in stamp
    assert hashlib.sha256(_json_bytes(stamp)).hexdigest() == (
        "44e086aef3dc7125dc1e6fae9b5cb6f60640a9e902d821ba158a418750336ce8"
    )
    restored = cb_serialization_context_from_stamp(stamp, where="test")
    assert restored.codebook_source_by_family is None
    assert cb_serialization_context_stamp(
        restored, formats=[FP4_FORMAT, FP8_FORMAT]
    ) == stamp

    breakdown = cb_assignment_payload_breakdown(
        {"fp4": FP4_FORMAT, "fp8": FP8_FORMAT},
        {"fp4": (8, 256), "fp8": (8, 256)},
        context=context,
    )
    assert "codebook_source_by_family" not in breakdown["context"]
    assert hashlib.sha256(_json_bytes(breakdown)).hexdigest() == (
        "a3037a3e5cc7528f58ca32d5cae24785a63a904a5e29bb4256a3b1a1ea8c2ec2"
    )

    config = _config(context, fp8_ref="lattice")
    assert "codebook_source_by_family" not in config["provenance"]
    assert hashlib.sha256(_json_bytes(config)).hexdigest() == (
        "2b075b5a67df4be9cd825fbed6924b71029555d7ce79f53664fa61586d6ace48"
    )


def test_keyed_setting_is_complete_and_fail_closed():
    assert parse_codebook_source_setting(
        "fp4=lattice,fp8=learned", where="test"
    ) == (
        "mixed",
        {"fp4": "lattice", "fp8": "learned"},
    )
    with pytest.raises(ValueError, match="must name every CB grid"):
        parse_codebook_source_setting("fp4=lattice", where="test")
    with pytest.raises(ValueError, match="names.*twice"):
        parse_codebook_source_setting(
            "fp4=lattice,fp4=learned,fp8=lattice", where="test"
        )
    with pytest.raises(ValueError, match="unknown CB grid"):
        parse_codebook_source_setting(
            "fp4=lattice,fp9=learned", where="test"
        )

    context_without_digests = CBSerializationContext.production(
        codebook_source_by_family={
            "fp4": "lattice",
            "fp8": "learned",
        }
    )
    with pytest.raises(ValueError, match="requires materialized"):
        cb_serialization_context_stamp(
            context_without_digests, formats=[FP4_FORMAT, FP8_FORMAT]
        )
    with pytest.raises(ValueError, match="exact materialized codebook"):
        cb_fields_for_context(
            fr.get_format(FP8_FORMAT),
            torch.zeros(2, 256),
            context=_mixed_context(),
        )


def test_mixed_stamp_payload_and_cost_proofs_are_per_family():
    context = _mixed_context()
    stamp = cb_serialization_context_stamp(
        context, formats=[FP4_FORMAT, FP8_FORMAT]
    )
    assert stamp["codebook_source"] == "mixed"
    assert stamp["codebook_source_by_family"] == {
        "fp4": "lattice",
        "fp8": "learned",
    }
    assert set(stamp["lattice_codebook_sha256_by_format"]) == {FP4_FORMAT}
    assert cb_serialization_context_stamp(
        cb_serialization_context_from_stamp(stamp, where="test"),
        formats=[FP4_FORMAT, FP8_FORMAT],
    ) == stamp

    breakdown = cb_assignment_payload_breakdown(
        {FP4_QNAME: FP4_FORMAT, FP8_QNAME: FP8_FORMAT},
        {FP4_QNAME: (8, 256), FP8_QNAME: (8, 256)},
        context=context,
    )
    assert breakdown["per_tensor"][FP4_QNAME]["sidecar_identity"][
        "codebook_source"
    ] == "lattice"
    assert breakdown["per_tensor"][FP8_QNAME]["sidecar_identity"][
        "codebook_source"
    ] == "learned"

    payload = {"provenance": {"cb_serialized_payload": stamp}}
    validate_cb_cost_provenance(
        payload,
        [FP4_FORMAT, FP8_FORMAT],
        context=context,
        where="test cost",
    )
    missing_lattice = copy.deepcopy(payload)
    del missing_lattice["provenance"]["cb_serialized_payload"][
        "lattice_codebook_sha256_by_format"
    ]
    with pytest.raises(ValueError, match="canonical lattice bytes"):
        validate_cb_cost_provenance(
            missing_lattice,
            [FP4_FORMAT, FP8_FORMAT],
            context=context,
            where="test cost",
        )

    swapped = copy.deepcopy(stamp)
    swapped["codebook_source_by_family"] = {
        "fp4": "learned",
        "fp8": "lattice",
    }
    with pytest.raises(ValueError, match="differs.*per family"):
        validate_cb_serialization_context_stamp(
            swapped, context, where="test stamp"
        )


def test_mixed_warm_projection_preserves_old_fp4_record_identity():
    legacy = warm_serialization_context(
        CBSerializationContext.production(), FP4_FORMAT
    )
    mixed = warm_serialization_context(_mixed_context(), FP4_FORMAT)
    assert mixed == legacy
    assert "codebook_source_by_family" not in mixed["serialization"]
    assert encoder_initializer_identity(_mixed_context(), FP4_FORMAT)[
        "codebook_source"
    ] == "lattice"
    with pytest.raises(ValueError, match="requires codebook_source='lattice'"):
        encoder_initializer_identity(_mixed_context(), FP8_FORMAT)


def test_mixed_render_identity_checks_both_source_arms():
    context = _mixed_context()
    formats = {
        FP4_QNAME: [FP4_FORMAT],
        FP8_QNAME: [FP8_FORMAT],
    }
    col_weights = {
        FP4_QNAME: torch.ones(256),
        FP8_QNAME: torch.ones(256),
    }
    source_weights = {
        FP4_QNAME: torch.zeros(2, 256),
        FP8_QNAME: torch.ones(2, 256),
    }
    identity = build_production_cache_cb_render_identity(
        formats,
        cb_serialization_context=context,
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    assert identity is not None
    identity = bind_cb_render_identity_source_weights(identity, source_weights)
    validate_cb_render_identity_metadata(
        identity,
        expected_context=context,
        expected_formats_by_qname=formats,
        col_weights=col_weights,
        source_weights=source_weights,
        where="test render",
    )

    missing_lattice = copy.deepcopy(identity)
    del missing_lattice["cb_serialized_payload"][
        "lattice_codebook_sha256_by_format"
    ]
    with pytest.raises(ValueError, match="canonical lattice codebook bytes"):
        validate_cb_render_identity_metadata(
            missing_lattice,
            expected_context=context,
            col_weights=col_weights,
            source_weights=source_weights,
            where="test render",
        )

    missing_learned = copy.deepcopy(identity)
    del missing_learned["cb_serialized_payload"][
        "codebook_content_sha256"
    ]
    with pytest.raises(ValueError, match="missing codebook_content_sha256"):
        validate_cb_render_identity_metadata(
            missing_learned,
            expected_context=context,
            col_weights=col_weights,
            source_weights=source_weights,
            where="test render",
        )


def test_config_groups_resolve_their_own_family_source():
    config = _config(_mixed_context(), fp8_ref="v_proj")
    schemes = {
        group["format"]: group["scheme"]
        for group in config["config_groups"].values()
    }
    assert schemes[FP4_FORMAT]["codebook_source"] == "lattice"
    assert schemes[FP8_FORMAT]["codebook_source"] == "learned"
    assert config["provenance"]["codebook_source"] == "mixed"
    assert config["provenance"]["codebook_source_by_family"] == {
        "fp4": "lattice",
        "fp8": "learned",
    }
