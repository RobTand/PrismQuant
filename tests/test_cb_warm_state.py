"""CPU-only tests for CB encoder warm-state plumbing.

Most fallback properties use a tiny deterministic fake.  Byte-identity uses
the real encoder, and record validation always uses the real serialization
context and safetensors sidecar implementation.
"""
from __future__ import annotations

import copy
import random

import pytest
import torch

from prismaquant.cb_warm_state import (
    CBEncodedPayload,
    CBWarmStartSession,
    CBWarmStateRecord,
    CBWarmStateStore,
    WarmStateVerificationError,
    build_warm_record,
    execute_warm_started_encode,
    selected_scale_state,
    tensor_value_identity,
    warm_serialization_context,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_serialization_context_stamp,
)


FORMAT = "NVFP4_CB_K12"
FP8_FORMAT = "FP8_CB_K28"
QNAME = "model.layers.0.self_attn.q_proj"


def _case(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(2, 256, generator=generator)
    col_weights = torch.rand(256, generator=generator) + 0.1
    # E=127, c=0 composes to exact E4M3 1.0 for every group.
    fields = {
        "scales": torch.ones(2, 16),
        "scale_super": torch.full((2, 1), 127, dtype=torch.uint8),
        "scale_sub": torch.zeros(2, 16, dtype=torch.uint8),
    }
    context = CBSerializationContext.production(encode_tier="balanced")
    record = build_warm_record(
        qname=QNAME,
        format_name=FORMAT,
        source_weight=weight,
        col_weights=col_weights,
        context=context,
        fields=fields,
    )
    return weight, col_weights, context, record


def _load(store, weight, col_weights, context, *, qname=QNAME, fmt=FORMAT):
    source_shape, source_digest = tensor_value_identity(weight)
    col_shape, col_digest = tensor_value_identity(col_weights)
    return store.load_matching(
        qname=qname,
        format_name=fmt,
        source_shape=source_shape,
        source_digest=source_digest,
        col_weights_shape=col_shape,
        col_weights_digest=col_digest,
        context=context,
    )


def _fake_payload(source: torch.Tensor, scales: torch.Tensor) -> CBEncodedPayload:
    # Assignment is deterministic given source + selected scale.  The output
    # tensor stands in for the exact packed byte stream.
    rendered = torch.remainder(
        torch.round(source * 17).to(torch.int64)
        + int(torch.round(scales.sum()).item()),
        256,
    ).to(torch.uint8)
    return CBEncodedPayload(
        value=rendered,
        selected_scale={"scales": scales.clone()},
        rendered={"packed": rendered},
    )


def _record_with_prepatch_global_stamp(
    record: CBWarmStateRecord,
    *,
    context: CBSerializationContext,
    format_name: str,
    drop_ldlq_scope: bool = False,
) -> CBWarmStateRecord:
    metadata = copy.deepcopy(dict(record.metadata))
    # This is the exact aggregate stamp the old warm writer obtained before
    # warm_serialization_context projected LDLQ onto its one format.
    serialization = cb_serialization_context_stamp(
        context, formats=[format_name]
    )
    if drop_ldlq_scope:
        serialization.pop("ldlq_scope")
    metadata["serialization_context"]["serialization"] = serialization
    return CBWarmStateRecord(metadata=metadata, scale_state=record.scale_state)


def test_warm_record_round_trip_is_atomic_and_content_keyed(tmp_path):
    weight, col_weights, context, record = _case()
    store = CBWarmStateStore(tmp_path)
    path = store.write(record)

    loaded = _load(store, weight, col_weights, context)
    assert loaded is not None
    assert loaded.metadata == record.metadata
    assert set(loaded.scale_state) == {"scales", "scale_super", "scale_sub"}
    assert all(
        torch.equal(loaded.scale_state[key], record.scale_state[key])
        for key in loaded.scale_state
    )
    assert path == store.path_for(
        QNAME, FORMAT, record.metadata["source_digest"]
    )
    assert not list(path.parent.glob("*.tmp"))

    other_qname = QNAME.replace("q_proj", "k_proj")
    other_path = store.path_for(
        other_qname, FORMAT, record.metadata["source_digest"]
    )
    changed_digest = tensor_value_identity(weight + 1)[1]
    changed_path = store.path_for(QNAME, FORMAT, changed_digest)
    assert len({path, other_path, changed_path}) == 3


def test_missing_scope_raw_stamp_canonicalizes_to_full_identity(tmp_path):
    weight, col_weights, context, record = _case()
    store = CBWarmStateStore(tmp_path)
    legacy = _record_with_prepatch_global_stamp(
        record,
        context=context,
        format_name=FORMAT,
        drop_ldlq_scope=True,
    )
    serialization = legacy.metadata["serialization_context"]["serialization"]
    assert serialization["ldlq"] is False
    assert "ldlq_scope" not in serialization
    assert "ldlq_packed_kernel" not in serialization
    store.write(legacy)

    loaded = _load(store, weight, col_weights, context)

    assert loaded is not None
    assert loaded.metadata == record.metadata


def test_prepatch_global_scope_verified_warm_encode_is_byte_identical(tmp_path):
    from prismaquant import nvfp4_cb_formats as cb

    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(2, 256, generator=generator) * 0.05
    col_weights = torch.rand(256, generator=generator) + 0.1
    # The campaign-wide scope enables LDLQ for NVFP4 only.  This FP8 cell's
    # canonical warm identity must therefore remain raw and independent of the
    # other family's setting.
    context = CBSerializationContext.production(
        encode_tier="fast", ldlq=True, ldlq_scope="nvfp4"
    )

    def encode(warm_scale_state=None):
        packed, fields = cb.nvfp4_cb_pack(
            weight,
            28,
            grid="fp8",
            mode="product",
            col_weights=col_weights,
            scale_sweep=True,
            scale_coding="v1",
            encode_tier="fast",
            warm_scale_state=warm_scale_state,
        )
        return CBEncodedPayload(
            value=(packed, fields),
            selected_scale=selected_scale_state(fields),
            rendered={"packed": packed, "weight_scale": fields["scales"]},
        )

    original_cold = encode()
    record = build_warm_record(
        qname=QNAME,
        format_name=FP8_FORMAT,
        source_weight=weight,
        col_weights=col_weights,
        context=context,
        fields=original_cold.value[1],
    )
    serialization = warm_serialization_context(context, FP8_FORMAT)[
        "serialization"
    ]
    assert serialization["ldlq"] is False
    assert serialization["ldlq_scope"] == "none"
    assert "ldlq_packed_kernel" not in serialization

    store = CBWarmStateStore(tmp_path)
    prepatch = _record_with_prepatch_global_stamp(
        record,
        context=context,
        format_name=FP8_FORMAT,
    )
    global_serialization = prepatch.metadata["serialization_context"][
        "serialization"
    ]
    assert global_serialization["ldlq"] is True
    assert global_serialization["ldlq_scope"] == "nvfp4"
    assert "ldlq_packed_kernel" in global_serialization
    store.write(prepatch)
    loaded = _load(
        store, weight, col_weights, context, fmt=FP8_FORMAT
    )
    assert loaded is not None
    assert loaded.metadata == record.metadata

    warm, outcome = execute_warm_started_encode(
        qname=QNAME,
        format_name=FP8_FORMAT,
        record=loaded,
        verify=True,
        full_encode=encode,
        seeded_encode=encode,
    )

    assert outcome == "verified"
    assert torch.equal(
        warm.rendered["packed"], original_cold.rendered["packed"]
    )
    assert torch.equal(
        warm.rendered["weight_scale"],
        original_cold.rendered["weight_scale"],
    )


def test_global_scope_canonicalization_rejects_semantic_and_kernel_drift(
    tmp_path,
):
    generator = torch.Generator().manual_seed(29)
    weight = torch.randn(2, 256, generator=generator) * 0.05
    col_weights = torch.rand(256, generator=generator) + 0.1
    context = CBSerializationContext.production(
        encode_tier="fast", ldlq=True, ldlq_scope="nvfp4"
    )
    record = build_warm_record(
        qname=QNAME,
        format_name=FP8_FORMAT,
        source_weight=weight,
        col_weights=col_weights,
        context=context,
        fields={"scales": torch.ones(2, 1)},
    )
    aggregate = _record_with_prepatch_global_stamp(
        record,
        context=context,
        format_name=FP8_FORMAT,
    )
    cases = {}

    inconsistent_bool = copy.deepcopy(aggregate)
    inconsistent_bool.metadata["serialization_context"]["serialization"][
        "ldlq"
    ] = False
    cases["scope-bool-inconsistent"] = inconsistent_bool

    invalid_scope = copy.deepcopy(aggregate)
    invalid_scope.metadata["serialization_context"]["serialization"][
        "ldlq_scope"
    ] = "fp8"
    cases["invalid-scope"] = invalid_scope

    wrong_kernel = copy.deepcopy(aggregate)
    wrong_kernel.metadata["serialization_context"]["serialization"][
        "ldlq_packed_kernel"
    ] = {"schema": "wrong"}
    cases["irrelevant-but-wrong-kernel"] = wrong_kernel

    missing_kernel = copy.deepcopy(aggregate)
    missing_kernel.metadata["serialization_context"]["serialization"].pop(
        "ldlq_packed_kernel"
    )
    cases["irrelevant-but-missing-kernel"] = missing_kernel

    active_fp8 = CBSerializationContext.production(
        encode_tier="fast", ldlq=True, ldlq_scope="all"
    )
    cases["different-effective-semantics"] = _record_with_prepatch_global_stamp(
        record,
        context=active_fp8,
        format_name=FP8_FORMAT,
    )

    for name, stale in cases.items():
        store = CBWarmStateStore(tmp_path / name)
        store.write(stale)
        assert _load(
            store, weight, col_weights, context, fmt=FP8_FORMAT
        ) is None, name


def test_source_digest_mismatch_refuses_record_and_falls_back(tmp_path):
    weight, col_weights, context, record = _case()
    store = CBWarmStateStore(tmp_path)
    store.write(record)
    assert _load(store, weight + 1, col_weights, context) is None

    session = CBWarmStartSession({}, all_qnames=[QNAME], verify_sample=32)
    calls = {"cold": 0, "warm": 0}
    full_scales = torch.tensor([3.0])

    def full():
        calls["cold"] += 1
        return _fake_payload(weight, full_scales)

    def seeded(state):
        calls["warm"] += 1
        return _fake_payload(weight, state["scales"])

    session.encode(QNAME, FORMAT, full_encode=full, seeded_encode=seeded)
    assert calls == {"cold": 1, "warm": 0}
    assert session.provenance() == {
        "warm_used": 0,
        "cold_fallback": 1,
        "verified_n": 0,
    }


def test_serialization_context_mismatch_refuses_record_and_falls_back(tmp_path):
    weight, col_weights, context, record = _case()
    store = CBWarmStateStore(tmp_path)
    store.write(record)
    changed = CBSerializationContext.production(encode_tier="fast")
    assert _load(store, weight, col_weights, changed) is None
    assert _load(store, weight, col_weights, context) is not None

    session = CBWarmStartSession({}, all_qnames=[QNAME], verify_sample=32)
    calls = {"cold": 0, "warm": 0}

    def full():
        calls["cold"] += 1
        return _fake_payload(weight, torch.tensor([3.0]))

    def seeded(state):
        calls["warm"] += 1
        return _fake_payload(weight, state["scales"])

    session.encode(QNAME, FORMAT, full_encode=full, seeded_encode=seeded)
    assert calls == {"cold": 1, "warm": 0}
    assert session.provenance()["cold_fallback"] == 1


def test_cost_encoder_persists_real_scale_argmin_cpu(tmp_path, monkeypatch):
    from prismaquant import format_registry as fr
    from prismaquant.measure_quant_cost import _cb_cost_quantize_dequantize

    warm_dir = tmp_path / "cost-warm"
    monkeypatch.setenv("PRISMAQUANT_CB_WARM_STATE_DIR", str(warm_dir))
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_SCALE_SWEEP", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "0")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "fast")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_COMPILE", "0")
    generator = torch.Generator().manual_seed(41)
    weight = torch.randn(2, 256, generator=generator) * 0.05
    col_weights = torch.rand(256, generator=generator) + 0.1

    rendered = _cb_cost_quantize_dequantize(
        fr.get_format(FORMAT),
        weight,
        col_weights=col_weights,
        qname=QNAME,
    )

    assert rendered.shape == weight.shape
    context = CBSerializationContext.production(encode_tier="fast")
    loaded = _load(
        CBWarmStateStore(warm_dir), weight, col_weights, context
    )
    assert loaded is not None
    assert set(loaded.scale_state) == {"scales", "scale_super", "scale_sub"}


def test_verify_sample_byte_mismatch_aborts():
    weight, _col_weights, _context, record = _case()
    cold_scale = record.scale_state["scales"]

    def full():
        return _fake_payload(weight, cold_scale)

    def bad_seeded(state):
        payload = _fake_payload(weight, state["scales"])
        bad = payload.rendered["packed"].clone()
        bad.view(-1)[0] ^= 1
        return CBEncodedPayload(
            value=bad,
            selected_scale=payload.selected_scale,
            rendered={"packed": bad},
        )

    with pytest.raises(
        WarmStateVerificationError,
        match="rendered byte mismatch.*never trusted",
    ):
        execute_warm_started_encode(
            qname=QNAME,
            format_name=FORMAT,
            record=record,
            verify=True,
            full_encode=full,
            seeded_encode=bad_seeded,
        )


def test_session_provenance_counts_warm_cold_and_verified():
    weight, _col_weights, _context, record = _case()
    names = [QNAME, QNAME.replace("q_proj", "k_proj"), "missing"]
    records = {
        names[0]: record,
        names[1]: CBWarmStateRecord(
            metadata={**record.metadata, "qname": names[1]},
            scale_state=record.scale_state,
        ),
    }
    session = CBWarmStartSession(records, all_qnames=names, verify_sample=1)

    def full():
        return _fake_payload(weight, record.scale_state["scales"])

    def seeded(state):
        return _fake_payload(weight, state["scales"])

    for name in names:
        session.encode(name, FORMAT, full_encode=full, seeded_encode=seeded)
    assert session.provenance() == {
        "warm_used": 2,
        "cold_fallback": 1,
        "verified_n": 1,
    }


@pytest.mark.parametrize("seed", range(16))
def test_deterministic_fake_warm_encode_equals_full_encode_property(seed):
    generator = random.Random(seed)
    source = torch.tensor(
        [generator.uniform(-3, 3) for _ in range(37)], dtype=torch.float32
    )
    scale = torch.tensor(
        [generator.choice((0.25, 0.5, 1.0, 2.0))], dtype=torch.float32
    )
    record = CBWarmStateRecord(
        metadata={"source_digest": f"{seed:064x}"},
        scale_state={"scales": scale},
    )
    cold = lambda: _fake_payload(source, scale)
    warm = lambda state: _fake_payload(source, state["scales"])
    payload, outcome = execute_warm_started_encode(
        qname=f"unit.{seed}",
        format_name=FORMAT,
        record=record,
        verify=True,
        full_encode=cold,
        seeded_encode=warm,
    )
    assert outcome == "verified"
    assert torch.equal(payload.value, cold().value)
