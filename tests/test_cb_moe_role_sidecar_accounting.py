from __future__ import annotations

import hashlib

import pytest

from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    codebook_sidecar_payload_bytes,
    codebook_subtable_shapes,
)


_FORMAT = "FP8_CBL_K28"


def _refs(label: str) -> tuple[str, ...]:
    return tuple(
        f"cb_codebook.{label}.{_FORMAT}.sub{index}"
        for index, _shape in enumerate(codebook_subtable_shapes(_FORMAT))
    )


def _learned_context(
    refs_by_qname: dict[str, tuple[str, ...]],
) -> CBSerializationContext:
    all_refs = {
        ref
        for refs in refs_by_qname.values()
        for ref in refs
    }
    return CBSerializationContext.production(
        codebook_source="learned",
        codebook_refs_by_qname_format={
            qname: {_FORMAT: refs}
            for qname, refs in refs_by_qname.items()
        },
        codebook_content_digests={
            ref: hashlib.sha256(ref.encode()).hexdigest()
            for ref in all_refs
        },
    )


def test_fused_routed_moe_accounts_gate_up_and_direct_down_sidecars():
    prefix = "model.layers.3.mlp.experts."
    fused = f"{prefix}gate_up_proj"
    down = f"{prefix}down_proj"
    gate_refs = _refs("layer3.gate")
    up_refs = _refs("layer3.up")
    down_refs = _refs("layer3.down")
    context = _learned_context({
        f"{prefix}gate_proj": gate_refs,
        f"{prefix}up_proj": up_refs,
        down: down_refs,
    })

    payload = cb_assignment_payload_breakdown(
        {fused: _FORMAT, down: _FORMAT},
        {fused: (4, 256, 256), down: (4, 128, 256)},
        context=context,
    )

    sidecar_bytes = codebook_sidecar_payload_bytes(_FORMAT)
    assert payload["codebook_sidecar_bytes"] == 3 * sidecar_bytes
    assert len(payload["sidecars"]) == 3
    assert {
        tuple(sidecar["codebook_ref"])
        for sidecar in payload["sidecars"]
    } == {gate_refs, up_refs, down_refs}
    fused_item = payload["per_tensor"][fused]
    assert fused_item["sidecar_payload_bytes"] == 2 * sidecar_bytes
    assert [
        tuple(sidecar["codebook_ref"])
        for sidecar in fused_item["sidecar_identities"]
    ] == [gate_refs, up_refs]
    assert "sidecar_identities" not in payload["per_tensor"][down]


def test_fused_routed_moe_role_sidecars_deduplicate_by_physical_refs():
    prefixes = [
        "model.layers.3.mlp.experts.",
        "model.layers.4.mlp.experts.",
    ]
    gate_refs = _refs("shared.gate")
    up_refs = _refs("shared.up")
    context = _learned_context({
        f"{prefix}{role}_proj": refs
        for prefix in prefixes
        for role, refs in (("gate", gate_refs), ("up", up_refs))
    })
    assignment = {
        f"{prefix}gate_up_proj": _FORMAT for prefix in prefixes
    }

    payload = cb_assignment_payload_breakdown(
        assignment,
        {qname: (2, 256, 256) for qname in assignment},
        context=context,
    )

    assert len(payload["sidecars"]) == 2
    assert payload["codebook_sidecar_bytes"] == (
        2 * codebook_sidecar_payload_bytes(_FORMAT)
    )


def test_fused_routed_moe_rejects_incomplete_logical_role_refs():
    prefix = "model.layers.3.mlp.experts."
    fused = f"{prefix}gate_up_proj"
    context = _learned_context({f"{prefix}gate_proj": _refs("gate")})

    with pytest.raises(ValueError, match="per-role sidecars are incomplete"):
        cb_assignment_payload_breakdown(
            {fused: _FORMAT},
            {fused: (2, 256, 256)},
            context=context,
        )


@pytest.mark.parametrize(
    "qname",
    [
        "model.layers.3.mlp.experts.gate_up_proj",
        "model.layers.3.mlp.gate_up_proj",
    ],
)
def test_no_logical_role_refs_preserves_single_fused_sidecar(qname: str):
    direct_refs = _refs("legacy.fused")
    context = _learned_context({qname: direct_refs})

    payload = cb_assignment_payload_breakdown(
        {qname: _FORMAT},
        {qname: (2, 256, 256)},
        context=context,
    )

    assert len(payload["sidecars"]) == 1
    assert payload["sidecars"][0]["codebook_ref"] == list(direct_refs)
    assert "sidecar_identities" not in payload["per_tensor"][qname]
