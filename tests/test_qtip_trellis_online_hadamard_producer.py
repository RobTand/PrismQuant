from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib
import json

import pytest
import torch

from prismaquant import format_registry
from prismaquant.trellis_formats import E2M1_FAMILY, native_code_value
from prismaquant.trellis_wire import TrellisWire, decode_values_torch


M = importlib.import_module(
    "research.qtip_native_nvfp4_2026-08-30."
    "trellis_online_hadamard_producer"
)
NATIVE = importlib.import_module(
    "research.qtip_native_nvfp4_2026-08-30.native_nvfp4_ldlq"
)


def _e2_alphabet(rate):
    ordered = tuple(sorted(
        range(16),
        key=lambda code: (native_code_value(E2M1_FAMILY, code), code),
    ))
    count = 1 << (rate + 1)
    if rate == 2:
        return (15, 13, 11, 9, 8, 2, 4, 7)
    return tuple(
        ordered[index * (len(ordered) - 1) // (count - 1)]
        for index in range(count)
    )


def _contract(rows: int = 8, columns: int = 16):
    return M.build_online_transform(
        rows=rows,
        columns=columns,
        input_block_size=4,
        output_block_size=4,
        input_seed=0x1234,
        output_seed=0x5678,
    )


def _block_hadamard(dimension: int, block_size: int) -> torch.Tensor:
    block = torch.ones(1, 1, dtype=torch.float32)
    while block.shape[0] < block_size:
        block = torch.cat((
            torch.cat((block, block), dim=1),
            torch.cat((block, -block), dim=1),
        ), dim=0)
    block *= block_size ** -0.5
    result = torch.zeros(dimension, dimension)
    for first in range(0, dimension, block_size):
        result[first:first + block_size, first:first + block_size] = block
    return result


def _fixture():
    generator = torch.Generator().manual_seed(20260830)
    weight = torch.randn(8, 16, generator=generator)
    activations = torch.randn(19, 16, generator=generator)
    hessian = activations.T @ activations + 0.25 * torch.eye(16)
    return weight, activations, hessian


def test_rank_two_diagonal_source_refuses_before_any_tensor_conversion():
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoTensorOperations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise AssertionError(f"malformed Hessian reached tensor op {func}")

    weight = torch.ones(1, 16, dtype=torch.bfloat16)
    malformed = torch.ones(16, 16, dtype=torch.bfloat16)
    with NoTensorOperations():
        with pytest.raises(ValueError, match="rank-one diagonal vector"):
            M._validated_positive_hessian_diagonal(malformed, dimension=16)
        with pytest.raises(ValueError, match="rank one over the input width"):
            NATIVE.qtip_native_arm_from_diagonal_hessian(weight, malformed)


def _rehashed_prepared(prepared, mutation):
    receipt = copy.deepcopy(prepared.receipt)
    receipt.pop("identity_sha256")
    mutation(receipt)
    receipt["identity_sha256"] = M._canonical_sha256(receipt)
    return M.PreparedOneLinear(
        transformed_weight=prepared.transformed_weight,
        transformed_hessian=prepared.transformed_hessian,
        online_transform=prepared.online_transform,
        receipt=receipt,
    )


def _rehashed_structured_prepared(prepared, mutation):
    receipt = copy.deepcopy(prepared.receipt)
    receipt.pop("identity_sha256")
    mutation(receipt)
    receipt["identity_sha256"] = M._canonical_sha256(receipt)
    return M.PreparedDiagonalHessianOneLinear(
        transformed_weight=prepared.transformed_weight,
        source_hessian_diagonal=prepared.source_hessian_diagonal,
        online_transform=prepared.online_transform,
        receipt=receipt,
    )


def test_metadata_matches_gridbook_pinned_conformance_vectors():
    contract = _contract()
    assert M.seeded_sign_digest(
        "input", 19, 0x0123456789ABCDEF
    ) == "c9850b2a7c2d365cdb23964ff33c6e934fd3dde47bedb07b35eb9ba8823f6368"
    assert contract["input"]["sign_sha256"] == \
        "11a6e83c049227716147570449f8a019853aa9400dae038c3fcc53727d056b01"
    assert contract["output"]["sign_sha256"] == \
        "1b16b1df538ba12dc3f97edbb85caa7050d46c148134290feba80f8236c83db9"
    assert contract["transform_sha256"] == \
        "3231ab8ed01b068864b261a369a9cfda3e3e59ffbabd5bd0b1bb854c1ec844b5"
    assert M.validate_online_transform(contract, rows=8, columns=16) == contract


def test_qtip_source_audit_constants_match_native_isolate_and_local_source(
    monkeypatch,
):
    assert M.QTIP_REPOSITORY == NATIVE.QTIP_REPOSITORY
    assert M.QTIP_PINNED_COMMIT == NATIVE.QTIP_PINNED_COMMIT
    assert M.QTIP_SOURCE_FILES == NATIVE.QTIP_SOURCE_FILES
    assert M.producer_source_sha256() == hashlib.sha256(
        M._PRODUCER_SOURCE_PATH.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(M, "_current_producer_source_sha256", lambda: "0" * 64)
    with pytest.raises(ValueError, match="source changed since module import"):
        M._require_producer_source_unchanged()
    monkeypatch.undo()
    monkeypatch.setattr(M, "encoder_source_sha256", lambda: "0" * 64)
    with pytest.raises(
        ValueError, match="encoder source changed since module import"
    ):
        M._require_encoder_source_unchanged()
    monkeypatch.undo()
    import prismaquant.trellis_encoder as encoder_module

    loaded_encoder_identity = encoder_module.encoder_source_sha256()
    monkeypatch.setattr(
        encoder_module, "_current_encoder_source_sha256", lambda: "0" * 64
    )
    assert encoder_module.encoder_source_sha256() == loaded_encoder_identity
    with pytest.raises(
        ValueError, match="encoder source changed since module import"
    ):
        M._require_encoder_source_unchanged()
    monkeypatch.undo()
    monkeypatch.setattr(M, "scale_grid_source_sha256", lambda: "0" * 64)
    with pytest.raises(
        ValueError, match="scale-grid source changed since module import"
    ):
        M._require_scale_grid_source_unchanged()


@pytest.mark.parametrize("field,value", [
    ("algorithm", "different"),
    ("normalization", "none"),
    ("padding", "zero"),
])
def test_metadata_refuses_semantic_or_digest_drift(field, value):
    contract = _contract()
    contract[field] = value
    with pytest.raises(ValueError, match=field):
        M.validate_online_transform(contract, rows=8, columns=16)


def test_metadata_refuses_seed_geometry_and_unknown_field_drift():
    contract = _contract()
    bad_seed = copy.deepcopy(contract)
    bad_seed["input"]["seed"] += 1
    with pytest.raises(ValueError, match="sign_sha256 does not bind"):
        M.validate_online_transform(bad_seed, rows=8, columns=16)

    bad_geometry = copy.deepcopy(contract)
    bad_geometry["output"]["dimension"] = 4
    with pytest.raises(ValueError, match="does not match"):
        M.validate_online_transform(bad_geometry, rows=8, columns=16)

    unknown = copy.deepcopy(contract)
    unknown["future_order"] = "guess"
    with pytest.raises(ValueError, match="unknown"):
        M.validate_online_transform(unknown, rows=8, columns=16)


def test_weight_and_hessian_basis_match_explicit_matrices():
    weight, _activations, hessian = _fixture()
    contract = _contract()
    input_signs = M.seeded_signs("input", 16, 0x1234)
    output_signs = M.seeded_signs("output", 8, 0x5678)
    h_in = _block_hadamard(16, 4)
    h_out = _block_hadamard(8, 4)
    r_in = h_in @ torch.diag(input_signs)
    r_out = h_out @ torch.diag(output_signs)

    transformed_weight, transformed_hessian = \
        M.transform_weight_and_hessian(weight, hessian, contract)
    assert torch.allclose(
        transformed_weight, r_out @ weight @ r_in.T, rtol=1e-6, atol=1e-6
    )
    assert torch.allclose(
        transformed_hessian, r_in @ hessian @ r_in.T,
        rtol=1e-6, atol=3e-6,
    )


def test_transformed_hessian_matches_transformed_activation_gram():
    weight, activations, _hessian = _fixture()
    hessian = activations.T @ activations
    contract = _contract()
    _weight_t, hessian_t = M.transform_weight_and_hessian(
        weight, hessian, contract
    )
    activations_t = M.transformed_activations(activations, contract)
    assert torch.allclose(
        hessian_t, activations_t.T @ activations_t, rtol=2e-6, atol=2e-5
    )


def test_gridbook_runtime_bf16_boundaries_are_reference_exact():
    _weight, activations, _hessian = _fixture()
    contract = _contract()
    input_signs = M.seeded_signs("input", 16, 0x1234)
    h_in = _block_hadamard(16, 4)
    expected_input = ((activations * input_signs) @ h_in).to(torch.bfloat16)
    assert torch.equal(
        M.transformed_activations(
            activations, contract, runtime_bf16_boundary=True
        ),
        expected_input,
    )

    output = torch.arange(-24, 24, dtype=torch.float32).reshape(6, 8) / 8
    output_signs = M.seeded_signs("output", 8, 0x5678)
    h_out = _block_hadamard(8, 4)
    expected_output = ((output @ h_out) * output_signs).to(torch.bfloat16)
    assert torch.equal(
        M.inverse_transformed_outputs(
            output, contract, runtime_bf16_boundary=True
        ),
        expected_output,
    )


def test_post_decode_serve_algebra_round_trip_and_quadratic_invariance():
    weight, activations, hessian = _fixture()
    contract = _contract()
    transformed_weight, transformed_hessian = \
        M.transform_weight_and_hessian(weight, hessian, contract)

    # Model a future exact Gridbook reference decoder's output without making
    # any claim that this tensor came from physical wire bytes.
    decoded_q = transformed_weight + torch.linspace(
        -0.02, 0.02, transformed_weight.numel()
    ).reshape_as(transformed_weight)
    result = M.verify_post_decode_serve_algebra(
        decoded_q, activations, contract
    )
    assert result["status"] == "post_decode_matrix_algebra_verified"
    assert result["wire_identity_verified"] is False

    original_q = M.decoded_weight_in_original_basis(decoded_q, contract)
    original_error = weight - original_q
    transformed_error = transformed_weight - decoded_q
    original_proxy = ((original_error @ hessian) * original_error).sum()
    transformed_proxy = (
        (transformed_error @ transformed_hessian) * transformed_error
    ).sum()
    assert torch.allclose(
        original_proxy, transformed_proxy, rtol=5e-6, atol=2e-5
    )


def test_scaffold_is_opt_in_unregistered_and_prepares_exact_wire_seam():
    weight, _activations, hessian = _fixture()
    before = set(format_registry.REGISTRY)
    with pytest.raises(ValueError, match="research_opt_in"):
        M.prepare_one_linear_scaffold(
            weight,
            hessian,
            body_rate_q256=512,
            input_block_size=4,
            output_block_size=4,
            input_seed=0x1234,
            output_seed=0x5678,
            research_opt_in="",
        )

    prepared = M.prepare_one_linear_scaffold(
        weight,
        hessian,
        body_rate_q256=512,
        input_block_size=4,
        output_block_size=4,
        input_seed=0x1234,
        output_seed=0x5678,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    assert set(format_registry.REGISTRY) == before
    assert prepared.receipt["status"] == \
        "prepared_exact_trellis_wire_seam_available"
    assert prepared.receipt["format_registry_entries_created"] == 0
    assert prepared.receipt["producer_eligible"] is False
    assert prepared.receipt["wire"] == {
        "schema": "gridbook.trellis.wire.v1",
        "family": "TCQ_E2M1_R256",
        "body_rate_q256": 512,
        "terminal_grid": "E2M1",
        "scale_contract": "group16_fp8_e4m3_0p5_bpw",
        "qtip_bitshift_wire_allowed": False,
        "wire_bytes": None,
        "wire_identity_sha256": None,
        "encoder_invoked": False,
        "decoder_invoked": False,
    }
    body = dict(prepared.receipt)
    identity = body.pop("identity_sha256")
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert identity == hashlib.sha256(encoded).hexdigest()
    assert prepared.receipt["wire_seam"]["available_repository_api"] == [
        "tail-biting Viterbi path planes",
        "gridbook.trellis.wire.v1 immutable byte packer",
        "same-byte canonical parser and reference decoder",
    ]


def test_combined_producer_emits_same_byte_wire_and_serve_round_trip():
    generator = torch.Generator().manual_seed(20260830)
    weight = torch.randn(1, 256, generator=generator)
    activations = torch.randn(5, 256, generator=generator)
    hessian = activations.T @ activations + 0.25 * torch.eye(256)
    prepared = M.prepare_one_linear_scaffold(
        weight,
        hessian,
        body_rate_q256=512,
        input_block_size=16,
        output_block_size=1,
        input_seed=0x1234,
        output_seed=0x5678,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    before = set(format_registry.REGISTRY)
    artifact = M.require_combined_wire_round_trip(
        prepared,
        activations,
        body_rate_q256=512,
        schedule=[2] * 256,
        layout="fixed_quota_per_256",
        alphabets={2: (15, 13, 11, 9, 8, 2, 4, 7)},
        scale_rule="static_6",
        sb_chunk=1,
        determinism_mode="on",
        tailbite_candidates=4,
        backend="eager",
        point_route="full",
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    assert set(format_registry.REGISTRY) == before
    assert artifact.receipt["status"] == \
        "physical_wire_and_serve_algebra_verified"
    assert artifact.receipt["serve_algebra"]["wire_identity_verified"] is True
    assert artifact.receipt["trellis"]["same_byte_reparse_verified"] is True
    assert artifact.receipt["trellis_objective"][
        "full_off_diagonal_blockldlq_applied"
    ] is False
    assert artifact.receipt["qtip_bitshift_wire_allowed"] is False
    assert artifact.receipt["producer_eligible"] is False
    assert hashlib.sha256(artifact.wire_bytes).hexdigest() == artifact.receipt[
        "trellis"
    ]["wire_identity_sha256"]


def test_combined_producer_requires_explicit_opt_in():
    generator = torch.Generator().manual_seed(9)
    weight = torch.randn(1, 256, generator=generator)
    hessian = torch.eye(256)
    prepared = M.prepare_one_linear_scaffold(
        weight,
        hessian,
        body_rate_q256=512,
        input_block_size=16,
        output_block_size=1,
        input_seed=1,
        output_seed=2,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    with pytest.raises(ValueError, match="research_opt_in"):
        M.require_combined_wire_round_trip(
            prepared,
            torch.randn(2, 256, generator=generator),
            body_rate_q256=512,
            schedule=[2] * 256,
            layout="fixed_quota_per_256",
            alphabets={2: (15, 13, 11, 9, 8, 2, 4, 7)},
            scale_rule="static_6",
            sb_chunk=1,
            determinism_mode="on",
            tailbite_candidates=4,
            backend="eager",
            point_route="full",
            research_opt_in="",
        )


def test_combined_producer_refuses_prepared_tensor_identity_drift():
    weight = torch.zeros(1, 256)
    hessian = torch.eye(256)
    prepared = M.prepare_one_linear_scaffold(
        weight,
        hessian,
        body_rate_q256=512,
        input_block_size=16,
        output_block_size=1,
        input_seed=1,
        output_seed=2,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    tampered = M.PreparedOneLinear(
        transformed_weight=prepared.transformed_weight + 1,
        transformed_hessian=prepared.transformed_hessian,
        online_transform=prepared.online_transform,
        receipt=prepared.receipt,
    )
    with pytest.raises(ValueError, match="transformed tensor identity mismatch"):
        M.require_combined_wire_round_trip(
            tampered,
            torch.zeros(1, 256),
            body_rate_q256=512,
            schedule=[2] * 256,
            layout="fixed_quota_per_256",
            alphabets={2: (15, 13, 11, 9, 8, 2, 4, 7)},
            scale_rule="static_6",
            sb_chunk=1,
            determinism_mode="on",
            tailbite_candidates=4,
            backend="eager",
            point_route="full",
            research_opt_in=M.RESEARCH_OPT_IN,
        )


def test_prepared_boundary_rejects_rehashed_unknown_and_semantic_mutations():
    prepared = M.prepare_one_linear_scaffold(
        torch.zeros(1, 256),
        torch.eye(256),
        body_rate_q256=512,
        input_block_size=16,
        output_block_size=1,
        input_seed=1,
        output_seed=2,
        research_opt_in=M.RESEARCH_OPT_IN,
    )

    mutations = [
        (
            "root unknown field",
            lambda body: body.__setitem__("future_semantics", True),
            "unknown=.*future_semantics",
        ),
        (
            "fixed status",
            lambda body: body.__setitem__("status", "trust_me"),
            "status mismatch",
        ),
        (
            "basis orientation",
            lambda body: body["basis"].__setitem__(
                "row_input", "x H_in D_in"
            ),
            "basis mismatch",
        ),
        (
            "source authority",
            lambda body: body["source"]["authority"].__setitem__(
                "reauthenticated_at_encode", True
            ),
            "source.authority mismatch",
        ),
        (
            "wire semantic",
            lambda body: body["wire"].__setitem__(
                "terminal_grid", "not_E2M1"
            ),
            "wire contract mismatch",
        ),
        (
            "nested wire unknown field",
            lambda body: body["wire"].__setitem__("future_wire", 1),
            "prepared receipt wire has .*unknown=.*future_wire",
        ),
        (
            "wire seam substitution",
            lambda body: body["wire_seam"][
                "excluded_substitutions"
            ].clear(),
            "wire_seam mismatch",
        ),
        (
            "production eligibility",
            lambda body: body.__setitem__("producer_eligible", True),
            "producer_eligible mismatch",
        ),
        (
            "JSON bool/int alias",
            lambda body: body.__setitem__(
                "format_registry_entries_created", False
            ),
            "format_registry_entries_created mismatch",
        ),
    ]
    for _label, mutation, match in mutations:
        tampered = _rehashed_prepared(prepared, mutation)
        with pytest.raises(ValueError, match=match):
            M._validate_prepared_one_linear(tampered, body_rate_q256=512)


@pytest.mark.parametrize("buffer_blocks", [1, 2, 3])
def test_buffered_blockldl_recurrence_matches_unbuffered_oracle(buffer_blocks):
    generator = torch.Generator().manual_seed(20260831)
    basis = torch.randn(6, 6, generator=generator)
    hessian = basis @ basis.T + 0.5 * torch.eye(6)
    feedback, diagonal = M.qtip_block_ldl_factors(hessian, block_size=2)
    assert torch.allclose(
        feedback,
        NATIVE.qtip_block_unit_lower(hessian, block_size=2),
        rtol=1e-6,
        atol=1e-6,
    )
    unit_lower = feedback.clone()
    for first in range(0, 6, 2):
        unit_lower[first:first + 2, first:first + 2] = torch.eye(2)
    assert torch.allclose(
        unit_lower @ torch.block_diag(*diagonal.unbind()) @ unit_lower.T,
        hessian,
        rtol=2e-5,
        atol=2e-5,
    )
    weight = torch.randn(3, 6, generator=generator)

    def terminal(_index, target):
        return torch.round(target * 4) / 4

    reference_q, reference_targets = M.reverse_block_feedback_reference(
        weight, feedback, terminal, block_size=2
    )
    buffered_q, buffered_targets = M.reverse_block_feedback_buffered(
        weight,
        feedback,
        terminal,
        block_size=2,
        buffer_blocks=buffer_blocks,
    )
    assert torch.equal(buffered_q, reference_q)
    assert all(
        torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)
        for actual, expected in zip(
            buffered_targets, reference_targets, strict=True
        )
    )


def test_blockldl_trellis_terminal_refuses_dense_D_claim():
    weight = torch.zeros(1, 512)
    prepared_inputs = (
        M.prepare_one_linear_scaffold(
            weight,
            torch.eye(512),
            body_rate_q256=512,
            input_block_size=16,
            output_block_size=1,
            input_seed=1,
            output_seed=2,
            research_opt_in=M.RESEARCH_OPT_IN,
        ),
        M.prepare_one_linear_diagonal_hessian_scaffold(
            weight,
            torch.linspace(0.5, 1.5, 512),
            body_rate_q256=512,
            input_block_size=256,
            output_block_size=1,
            input_seed=1,
            output_seed=2,
            research_opt_in=M.RESEARCH_OPT_IN,
        ),
    )
    for prepared in prepared_inputs:
        with pytest.raises(ValueError, match="dense_block_D is unsupported"):
            M.require_blockldl_trellis_wire_round_trip(
                prepared,
                torch.zeros(1, 512),
                body_rate_q256=512,
                schedule=[2] * 512,
                layout="fixed_quota_per_256",
                alphabets={2: (15, 13, 11, 9, 8, 2, 4, 7)},
                scale_rule="static_6",
                sb_chunk=1,
                determinism_mode="on",
                tailbite_candidates=4,
                backend="eager",
                point_route="full",
                terminal_metric_mode="dense_block_D",
                buffer_blocks=1,
                research_opt_in=M.RESEARCH_OPT_IN,
            )


def test_blockldl_factorization_refuses_non_positive_definite_hessian():
    hessian = torch.eye(256)
    hessian[-1, -1] = -1
    with pytest.raises(ValueError, match="positive definite"):
        M.qtip_block_ldl_factors(hessian)


@pytest.mark.parametrize(("terminal_metric_mode", "buffer_blocks"), [
    ("diag_block_D", 1),
    ("qtip_frobenius", 2),
])
def test_blockldl_trellis_uses_full_feedback_and_one_same_byte_wire(
    terminal_metric_mode,
    buffer_blocks,
    monkeypatch,
):
    generator = torch.Generator().manual_seed(20260901)
    weight = torch.randn(2, 512, generator=generator)
    activations = torch.randn(9, 512, generator=generator)
    hessian = activations.T @ activations + 0.5 * torch.eye(512)
    prepared = M.prepare_one_linear_scaffold(
        weight,
        hessian,
        body_rate_q256=512,
        input_block_size=16,
        output_block_size=2,
        input_seed=0x1234,
        output_seed=0x5678,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    source_rechecks = 0
    original_source_recheck = M._require_implementation_sources_unchanged

    def tracked_source_recheck():
        nonlocal source_rechecks
        source_rechecks += 1
        return original_source_recheck()

    monkeypatch.setattr(
        M, "_require_implementation_sources_unchanged", tracked_source_recheck
    )
    before = set(format_registry.REGISTRY)
    artifact = M.require_blockldl_trellis_wire_round_trip(
        prepared,
        activations,
        body_rate_q256=512,
        schedule=[2] * 512,
        layout="fixed_quota_per_256",
        alphabets={2: (15, 13, 11, 9, 8, 2, 4, 7)},
        scale_rule="static_6",
        sb_chunk=2,
        determinism_mode="on",
        tailbite_candidates=4,
        backend="eager",
        point_route="full",
        terminal_metric_mode=terminal_metric_mode,
        buffer_blocks=buffer_blocks,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    assert set(format_registry.REGISTRY) == before
    wire = TrellisWire.from_bytes(artifact.wire_bytes)
    assert wire.to_bytes() == artifact.wire_bytes
    assert wire.columns == 512 and wire.rows == 2
    assert torch.equal(
        artifact.decoded_transformed_weight,
        decode_values_torch(artifact.wire_bytes),
    )
    receipt = artifact.receipt
    assert receipt["schema"] == M.BLOCKLDL_COMBINED_ARTIFACT_SCHEMA
    assert receipt["block_ldl"][
        "full_cross_block_feedback_matrix_consumed"
    ] is True
    expected_consumption = {
        "diagonal_consumed": terminal_metric_mode == "diag_block_D",
        "off_diagonal_consumed": False,
        "full_matrix_consumed": False,
        "exact_dense_objective": False,
    }
    assert receipt["block_ldl"][
        "dense_D_terminal_consumption"
    ] == expected_consumption
    assert all(
        block["dense_D_terminal_consumption"] == expected_consumption
        for block in receipt["terminal_blocks"]
    )
    assert receipt["block_ldl"][
        "terminal_metric_mode"
    ] == terminal_metric_mode
    assert receipt["block_ldl"]["block_size"] == 256
    assert receipt["block_ldl"]["buffer_blocks"] == buffer_blocks
    assert receipt["same_byte_reparse_verified"] is True
    assert receipt["serve_algebra"]["wire_identity_verified"] is True
    assert receipt["producer_eligible"] is False
    assert source_rechecks == 2
    provenance = receipt["implementation_provenance"]
    assert provenance["producer_source"] == {
        "path": (
            "research/qtip_native_nvfp4_2026-08-30/"
            "trellis_online_hadamard_producer.py"
        ),
        "sha256": M.producer_source_sha256(),
    }
    assert provenance["encoder_source"] == {
        "path": "prismaquant/trellis_encoder.py",
        "sha256": M._IMPORTED_ENCODER_SOURCE_SHA256,
    }
    assert provenance["qtip_source_audit"] == {
        "repository": M.QTIP_REPOSITORY,
        "commit": M.QTIP_PINNED_COMMIT,
        "source_sha256": dict(sorted(M.QTIP_SOURCE_FILES.items())),
        "runtime_or_wire_imported": False,
    }
    first_target = receipt["terminal_blocks"][0]["feedback_target_sha256"]
    assert first_target != M._tensor_sha256(
        prepared.transformed_weight[:, :256]
    )


@pytest.mark.parametrize(("layout", "schedule"), [
    (
        "fixed_quota_per_256",
        [2] * 256 + [1] * 128 + [3] * 128,
    ),
    (
        "tight_offsets",
        [1] * 144 + [2] * 112 + [2] * 112 + [3] * 144,
    ),
])
def test_blockldl_trellis_uses_block_local_recipe_and_full_union(
    layout,
    schedule,
):
    generator = torch.Generator().manual_seed(20260902)
    weight = torch.randn(1, 512, generator=generator)
    activations = torch.randn(5, 512, generator=generator)
    prepared = M.prepare_one_linear_scaffold(
        weight,
        activations.T @ activations + 0.5 * torch.eye(512),
        body_rate_q256=512,
        input_block_size=16,
        output_block_size=1,
        input_seed=3,
        output_seed=4,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    artifact = M.require_blockldl_trellis_wire_round_trip(
        prepared,
        activations,
        body_rate_q256=512,
        schedule=schedule,
        layout=layout,
        alphabets={rate: _e2_alphabet(rate) for rate in (1, 2, 3)},
        scale_rule="static_6",
        sb_chunk=1,
        determinism_mode="on",
        tailbite_candidates=4,
        backend="eager",
        point_route="full",
        terminal_metric_mode="diag_block_D",
        buffer_blocks=1,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    wire = TrellisWire.from_bytes(artifact.wire_bytes)
    assert wire.schedule == tuple(schedule)
    assert set(wire.alphabets) == {1, 2, 3}
    assert [
        block["local_body_rate_q256"]
        for block in artifact.receipt["terminal_blocks"]
    ] == [sum(schedule[:256]), sum(schedule[256:])]
    assert torch.equal(
        artifact.decoded_transformed_weight,
        decode_values_torch(artifact.wire_bytes),
    )


def test_two_transform_block_diagonal_path_matches_dense_factors_and_wire():
    generator = torch.Generator().manual_seed(20260903)
    weight = torch.randn(2, 1024, generator=generator)
    activations = torch.randn(3, 1024, generator=generator)
    diagonal = torch.rand(1024, generator=generator).add_(0.25)
    prepare_kwargs = {
        "body_rate_q256": 512,
        "input_block_size": 512,
        "output_block_size": 2,
        "input_seed": 0xABCDEF,
        "output_seed": 0x123456,
        "research_opt_in": M.RESEARCH_OPT_IN,
    }
    dense = M.prepare_one_linear_scaffold(
        weight, torch.diag(diagonal), **prepare_kwargs
    )
    structured = M.prepare_one_linear_diagonal_hessian_scaffold(
        weight, diagonal, **prepare_kwargs
    )
    assert torch.equal(dense.transformed_weight, structured.transformed_weight)

    groups = list(M.iter_transformed_diagonal_block_ldl_factors(
        diagonal, structured.online_transform
    ))
    assert [(g.first_column, g.last_column_exclusive) for g in groups] == [
        (0, 512), (512, 1024)
    ]
    assembled_hessian = torch.block_diag(
        *(group.transformed_hessian for group in groups)
    )
    assert torch.equal(assembled_hessian, dense.transformed_hessian)
    dense_feedback, dense_d = M.qtip_block_ldl_factors(
        dense.transformed_hessian
    )
    structured_feedback = torch.block_diag(
        *(group.feedback_lower for group in groups)
    )
    structured_d = torch.cat(
        [group.diagonal_blocks for group in groups], dim=0
    )
    assert torch.allclose(
        structured_feedback, dense_feedback, rtol=2e-6, atol=2e-7
    )
    assert torch.allclose(structured_d, dense_d, rtol=2e-6, atol=2e-7)
    assert torch.count_nonzero(dense_feedback[:512, 512:]) == 0
    assert torch.count_nonzero(dense_feedback[512:, :512]) == 0

    def terminal(_index, target):
        return torch.round(target * 8.0) / 8.0

    dense_q, dense_targets = M.reverse_block_feedback_buffered(
        dense.transformed_weight,
        dense_feedback,
        terminal,
        buffer_blocks=2,
    )
    structured_q = torch.zeros_like(dense_q)
    structured_targets = []
    for group in groups:
        first, last = group.first_column, group.last_column_exclusive
        group_q, group_targets = M.reverse_block_feedback_buffered(
            structured.transformed_weight[:, first:last],
            group.feedback_lower,
            terminal,
            buffer_blocks=2,
        )
        structured_q[:, first:last] = group_q
        structured_targets.extend(group_targets)
    assert torch.equal(structured_q, dense_q)
    assert all(
        torch.allclose(actual, expected, rtol=2e-6, atol=2e-7)
        for actual, expected in zip(
            structured_targets, dense_targets, strict=True
        )
    )

    encode_kwargs = {
        "activations": activations,
        "body_rate_q256": 512,
        "schedule": [2] * 1024,
        "layout": "fixed_quota_per_256",
        "alphabets": {2: _e2_alphabet(2)},
        "scale_rule": "static_6",
        "sb_chunk": 2,
        "determinism_mode": "on",
        "tailbite_candidates": 4,
        "backend": "eager",
        "point_route": "full",
        "terminal_metric_mode": "diag_block_D",
        "buffer_blocks": 2,
        "research_opt_in": M.RESEARCH_OPT_IN,
    }
    dense_artifact = M.require_blockldl_trellis_wire_round_trip(
        dense, **encode_kwargs
    )
    structured_artifact = M.require_blockldl_trellis_wire_round_trip(
        structured, **encode_kwargs
    )
    assert structured_artifact.wire_bytes == dense_artifact.wire_bytes
    assert torch.equal(
        structured_artifact.decoded_codes, dense_artifact.decoded_codes
    )
    assert torch.equal(
        structured_artifact.decoded_transformed_weight,
        dense_artifact.decoded_transformed_weight,
    )
    for field in (
        "wire_bytes", "wire_identity_sha256", "decoded_codes_sha256",
        "decoded_weight_sha256", "same_byte_reparse_verified",
    ):
        assert structured_artifact.receipt[field] == dense_artifact.receipt[field]
    structured_factor = structured_artifact.receipt["block_ldl"]
    assert structured_factor["factorization_strategy"] == (
        "exact_block_diagonal_from_retained_source_diagonal_v1"
    )
    assert structured_factor["dense_k_by_k_materialized"] is False
    assert structured_factor["factor_group_count"] == 2
    assert structured_factor["largest_factor_group_columns"] == 512
    assert structured_factor["full_cross_output_rows_processed"] is True
    assert structured_factor["cross_block_feedback_nonzero_count"] > 0
    assert structured_factor["dense_D_terminal_consumption"] == {
        "diagonal_consumed": True,
        "off_diagonal_consumed": False,
        "full_matrix_consumed": False,
        "exact_dense_objective": False,
    }
    structure = structured.receipt["transformed"]["hessian_structure"]
    assert [
        (item["index"], item["first_column"], item["last_column_exclusive"])
        for item in structure["ordered_blocks"]
    ] == [(0, 0, 512), (1, 512, 1024)]
    for expected, actual in zip(
        structure["ordered_blocks"],
        structured_factor["factor_groups"],
        strict=True,
    ):
        assert actual["index"] == expected["index"]
        assert actual["first_column"] == expected["first_column"]
        assert actual["last_column_exclusive"] == expected[
            "last_column_exclusive"
        ]
        assert actual["columns"] == expected["columns"]
        assert actual["source_diagonal_sha256"] == expected[
            "source_diagonal_sha256"
        ]


def test_arm_e_scale_grid_refuses_any_scope_inside_a_factor_group():
    weight = torch.randn(1, 512, generator=torch.Generator().manual_seed(711))
    prepared = M.prepare_one_linear_diagonal_hessian_scaffold(
        weight,
        torch.ones(512),
        body_rate_q256=512,
        input_block_size=512,
        output_block_size=1,
        input_seed=7,
        output_seed=11,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    with pytest.raises(ValueError, match="row_factor_group"):
        M.require_blockldl_trellis_wire_round_trip(
            prepared,
            torch.zeros(1, 512),
            body_rate_q256=512,
            schedule=[2] * 512,
            layout="fixed_quota_per_256",
            alphabets={2: _e2_alphabet(2)},
            scale_rule="static_6",
            sb_chunk=1,
            determinism_mode="on",
            tailbite_candidates=4,
            backend="eager",
            point_route="full",
            terminal_metric_mode="diag_block_D",
            buffer_blocks=1,
            research_opt_in=M.RESEARCH_OPT_IN,
            scale_grid_multipliers=(1.0, 0.75, 1.25),
            scale_grid_selection_scope="row_superblock",
        )
    for menu, message in (
        ((), "candidate zero"),
        ((1.0, 1.0), "unique"),
        ((1.0, float("nan")), "finite/positive"),
        ((True, 0.75), "plain numeric"),
        ((1.0, "0.75"), "plain numeric"),
    ):
        with pytest.raises(ValueError, match=message):
            M.require_blockldl_trellis_wire_round_trip(
                prepared,
                torch.zeros(1, 512),
                body_rate_q256=512,
                schedule=[2] * 512,
                layout="fixed_quota_per_256",
                alphabets={2: _e2_alphabet(2)},
                scale_rule="static_6",
                sb_chunk=1,
                determinism_mode="on",
                tailbite_candidates=4,
                backend="eager",
                point_route="full",
                terminal_metric_mode="diag_block_D",
                buffer_blocks=1,
                research_opt_in=M.RESEARCH_OPT_IN,
                scale_grid_multipliers=menu,
                scale_grid_selection_scope="row_factor_group",
            )


def test_arm_e_render_recipe_uses_one_immutable_schedule_snapshot():
    identity_schedule = [1] * 16 + [4] * 240
    later_schedule = [4] * 240 + [1] * 16

    class FlipSchedule(list):
        def __init__(self):
            super().__init__(identity_schedule)
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            selected = identity_schedule if self.iterations == 1 else later_schedule
            return iter(selected)

    generator = torch.Generator().manual_seed(20260915)
    weight = torch.randn(1, 256, generator=generator)
    prepared = M.prepare_one_linear_diagonal_hessian_scaffold(
        weight,
        torch.ones(256),
        body_rate_q256=976,
        input_block_size=256,
        output_block_size=1,
        input_seed=7,
        output_seed=11,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    schedule = FlipSchedule()
    artifact = M.require_blockldl_trellis_wire_round_trip(
        prepared,
        torch.zeros(1, 256),
        body_rate_q256=976,
        schedule=schedule,
        layout="tight_offsets",
        alphabets={1: (15, 11, 8, 4)},
        scale_rule="static_6",
        sb_chunk=1,
        determinism_mode="on",
        tailbite_candidates=4,
        backend="eager",
        point_route="full",
        terminal_metric_mode="diag_block_D",
        buffer_blocks=1,
        research_opt_in=M.RESEARCH_OPT_IN,
        scale_grid_multipliers=(1.0, 0.75, 1.25),
        scale_grid_selection_scope="row_factor_group",
    )
    wire = TrellisWire.from_bytes(artifact.wire_bytes)
    assert schedule.iterations == 1
    assert list(wire.schedule) == identity_schedule
    assert artifact.receipt["wire_recipe"]["schedule"] == identity_schedule


def test_arm_e_render_recipe_refuses_string_subclass_alias():
    class BackendAlias(str):
        def __str__(self):
            return "triton"

    with pytest.raises(ValueError, match="backend must be a nonempty plain string"):
        M.require_blockldl_trellis_wire_round_trip(
            None,
            torch.zeros(1, 256),
            body_rate_q256=976,
            schedule=[1] * 16 + [4] * 240,
            layout="tight_offsets",
            alphabets={1: (15, 11, 8, 4)},
            scale_rule="static_6",
            sb_chunk=1,
            determinism_mode="on",
            tailbite_candidates=4,
            backend=BackendAlias("eager"),
            point_route="full",
            terminal_metric_mode="diag_block_D",
            buffer_blocks=1,
            research_opt_in=M.RESEARCH_OPT_IN,
        )


def test_arm_e_identity_grid_runs_two_full_recurrences_and_is_old_bytes(
    monkeypatch,
):
    generator = torch.Generator().manual_seed(20260912)
    weight = torch.randn(2, 512, generator=generator)
    activations = torch.randn(5, 512, generator=generator)
    prepared = M.prepare_one_linear_scaffold(
        weight,
        activations.T @ activations + 0.5 * torch.eye(512),
        body_rate_q256=512,
        input_block_size=16,
        output_block_size=2,
        input_seed=3,
        output_seed=4,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    kwargs = {
        "activations": activations,
        "body_rate_q256": 512,
        "schedule": [2] * 512,
        "layout": "fixed_quota_per_256",
        "alphabets": {2: _e2_alphabet(2)},
        "scale_rule": "static_6",
        "sb_chunk": 2,
        "determinism_mode": "on",
        "tailbite_candidates": 4,
        "backend": "eager",
        "point_route": "full",
        "terminal_metric_mode": "diag_block_D",
        "buffer_blocks": 1,
        "research_opt_in": M.RESEARCH_OPT_IN,
    }
    identity = M.require_blockldl_trellis_wire_round_trip(prepared, **kwargs)
    calls = 0
    reference_calls = 0
    original = M.reverse_block_feedback_buffered
    original_reference = M.reverse_block_feedback_reference

    def counted(*args, **call_kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **call_kwargs)

    def counted_reference(*args, **call_kwargs):
        nonlocal reference_calls
        reference_calls += 1
        return original_reference(*args, **call_kwargs)

    monkeypatch.setattr(M, "reverse_block_feedback_buffered", counted)
    monkeypatch.setattr(
        M, "reverse_block_feedback_reference", counted_reference
    )
    gated = M.require_blockldl_trellis_wire_round_trip(
        prepared,
        **kwargs,
        scale_grid_multipliers=(1.0,),
        scale_grid_selection_scope="row_factor_group",
    )
    assert calls == 2
    assert reference_calls == 3
    assert gated.wire_bytes == identity.wire_bytes
    selection = gated.receipt["block_ldl"]["scale_selection"]
    assert selection["full_recurrence_arms"] == 2
    assert selection["selected_recurrence_reverified"] is True
    assert selection["candidate_win_rows"] == 0
    assert selection["no_win_byte_identical"] is True
    assert selection["wire_byte_delta"] == selection["delta_bpw_q256"] == 0
    assert selection["identity_wire_bytes"] == selection["final_wire_bytes"]
    assert selection["identity_wire_sha256"] == hashlib.sha256(
        identity.wire_bytes
    ).hexdigest()
    assert selection["final_wire_sha256"] == hashlib.sha256(
        gated.wire_bytes
    ).hexdigest()
    forged_receipt = copy.deepcopy(gated.receipt)
    forged_receipt["block_ldl"]["scale_selection"][
        "identity_wire_sha256"
    ] = "0" * 64
    forged_body = dict(forged_receipt)
    forged_body.pop("identity_sha256")
    forged_receipt["identity_sha256"] = M._canonical_sha256(forged_body)
    forged = replace(gated, receipt=forged_receipt)
    with pytest.raises(ValueError, match="receipt semantics"):
        M.require_blockldl_trellis_artifact_replay(
            forged,
            prepared,
            **kwargs,
            scale_grid_multipliers=(1.0,),
            scale_grid_selection_scope="row_factor_group",
        )


def test_arm_e_scale_grid_gates_one_mask_across_every_coupled_block():
    generator = torch.Generator().manual_seed(20260911)
    weight = torch.randn(2, 512, generator=generator)
    activations = torch.randn(6, 512, generator=generator)
    prepared = M.prepare_one_linear_scaffold(
        weight,
        activations.T @ activations + 0.5 * torch.eye(512),
        body_rate_q256=512,
        input_block_size=16,
        output_block_size=2,
        input_seed=3,
        output_seed=4,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    artifact = M.require_blockldl_trellis_wire_round_trip(
        prepared,
        activations,
        body_rate_q256=512,
        schedule=[2] * 512,
        layout="fixed_quota_per_256",
        alphabets={2: _e2_alphabet(2)},
        scale_rule="static_6",
        sb_chunk=2,
        determinism_mode="on",
        tailbite_candidates=4,
        backend="eager",
        point_route="full",
        terminal_metric_mode="diag_block_D",
        buffer_blocks=1,
        research_opt_in=M.RESEARCH_OPT_IN,
        scale_grid_multipliers=(1.0, 0.55, 0.75, 1.25, 1.30),
        scale_grid_selection_scope="row_factor_group",
    )
    selection = artifact.receipt["block_ldl"]["scale_selection"]
    assert selection["candidate_win_rows"] == 1
    assert selection["cf_exact_minimum_per_row_factor_group"] is True
    assert selection["cf_le_c0"] is True
    group = artifact.receipt["block_ldl"]["factor_groups"][0]
    assert group["scale_selection"]["full_recurrence_arms"] == 2
    assert group["scale_selection"]["candidate_win_rows"] == 1
    assert group["scale_selection"]["cf_exact_minimum"] is True
    # Both 256-column terminals are coupled by one recurrence and therefore
    # carry the same row/factor-group decision, never per-block decisions.
    terminal_selections = [
        block["scale_selection"] for block in artifact.receipt["terminal_blocks"]
    ]
    assert {item["scope"] for item in terminal_selections} == {
        "row_factor_group"
    }
    assert {item["candidate_win_rows"] for item in terminal_selections} == {1}
    assert artifact.receipt["wire_recipe"]["scale_selection"]["mode"] == (
        "e4m3_grid_gated_v1"
    )
    assert TrellisWire.from_bytes(artifact.wire_bytes).to_bytes() == (
        artifact.wire_bytes
    )


def test_arm_e_selected_recurrence_target_mutation_refuses(monkeypatch):
    generator = torch.Generator().manual_seed(20260916)
    weight = torch.randn(1, 256, generator=generator)
    prepared = M.prepare_one_linear_diagonal_hessian_scaffold(
        weight,
        torch.ones(256),
        body_rate_q256=512,
        input_block_size=256,
        output_block_size=1,
        input_seed=7,
        output_seed=11,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    original = M.reverse_block_feedback_reference
    calls = 0

    def corrupt_selected(*args, **kwargs):
        nonlocal calls
        calls += 1
        decoded, targets = original(*args, **kwargs)
        if calls == 3:
            targets = (targets[0] + 1.0, *targets[1:])
        return decoded, targets

    monkeypatch.setattr(M, "reverse_block_feedback_reference", corrupt_selected)
    with pytest.raises(AssertionError, match="selected Arm E targets differ"):
        M.require_blockldl_trellis_wire_round_trip(
            prepared,
            torch.zeros(1, 256),
            body_rate_q256=512,
            schedule=[2] * 256,
            layout="fixed_quota_per_256",
            alphabets={2: _e2_alphabet(2)},
            scale_rule="static_6",
            sb_chunk=1,
            determinism_mode="on",
            tailbite_candidates=4,
            backend="eager",
            point_route="full",
            terminal_metric_mode="diag_block_D",
            buffer_blocks=1,
            research_opt_in=M.RESEARCH_OPT_IN,
            scale_grid_multipliers=(1.0,),
            scale_grid_selection_scope="row_factor_group",
        )


def test_structured_scale_grid_respects_factor_boundaries_and_replays_exactly():
    generator = torch.Generator().manual_seed(0)
    weight = (
        torch.randn(2, 1024, generator=generator)
        * torch.exp(torch.linspace(-2.0, 2.0, 1024))[None, :]
    )
    importance = torch.rand(1024, generator=generator).add_(0.25)
    activations = torch.randn(2, 1024, generator=generator)
    prepared = M.prepare_one_linear_diagonal_hessian_scaffold(
        weight,
        importance,
        body_rate_q256=976,
        input_block_size=512,
        output_block_size=2,
        input_seed=3,
        output_seed=4,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    superblock_schedule = [1] * 16 + [4] * 240
    kwargs = {
        "activations": activations,
        "body_rate_q256": 976,
        "schedule": superblock_schedule * 4,
        "layout": "tight_offsets",
        "alphabets": {1: (15, 11, 8, 4)},
        "scale_rule": "static_6",
        "sb_chunk": 2,
        "determinism_mode": "on",
        "tailbite_candidates": 4,
        "backend": "eager",
        "point_route": "full",
        "terminal_metric_mode": "diag_block_D",
        "buffer_blocks": 1,
        "research_opt_in": M.RESEARCH_OPT_IN,
        "scale_grid_multipliers": (1.0, 0.55, 0.70, 0.85, 1.15, 1.30),
        "scale_grid_selection_scope": "row_factor_group",
    }
    first = M.require_blockldl_trellis_wire_round_trip(prepared, **kwargs)
    second = M.require_blockldl_trellis_artifact_replay(
        first, prepared, **kwargs
    )
    assert first.wire_bytes == second.wire_bytes
    assert first.receipt == second.receipt
    groups = first.receipt["block_ldl"]["factor_groups"]
    assert [(group["first_column"], group["last_column_exclusive"]) for group in groups] == [
        (0, 512), (512, 1024),
    ]
    selections = [group["scale_selection"] for group in groups]
    assert [item["candidate_win_rows"] for item in selections] == [2, 1]
    assert selections[0]["candidate_win_mask_sha256"] != selections[1][
        "candidate_win_mask_sha256"
    ]
    terminal_blocks = first.receipt["terminal_blocks"]
    assert [
        block["scale_selection"]["candidate_win_rows"]
        for block in terminal_blocks
    ] == [2, 2, 1, 1]
    assert [block["factor_group_index"] for block in terminal_blocks] == [0, 0, 1, 1]


@pytest.mark.parametrize("block_size", [128, 768, 2048])
def test_structured_diagonal_contract_refuses_invalid_transform_blocks(block_size):
    weight = torch.zeros(1, 1024)
    diagonal = torch.ones(1024)
    with pytest.raises(ValueError, match=(
        "multiple of 256|positive power of two|must divide"
    )):
        M.prepare_one_linear_diagonal_hessian_scaffold(
            weight,
            diagonal,
            body_rate_q256=512,
            input_block_size=block_size,
            output_block_size=1,
            input_seed=1,
            output_seed=2,
            research_opt_in=M.RESEARCH_OPT_IN,
        )


def test_structured_diagonal_contract_refuses_off_block_rank2_nonpositive_and_forgery():
    weight = torch.zeros(1, 512)
    diagonal = torch.linspace(0.5, 1.5, 512)
    dense = torch.diag(diagonal)
    dense[0, 300] = dense[300, 0] = 0.25
    with pytest.raises(ValueError, match="dense or off-diagonal"):
        M.prepare_one_linear_diagonal_hessian_scaffold(
            weight,
            dense,
            body_rate_q256=512,
            input_block_size=256,
            output_block_size=1,
            input_seed=1,
            output_seed=2,
            research_opt_in=M.RESEARCH_OPT_IN,
        )
    for bad in (diagonal.clone().zero_(), diagonal.clone()):
        if bool((bad > 0).all()):
            bad[17] = -1
        with pytest.raises(ValueError, match="strictly positive"):
            M.prepare_one_linear_diagonal_hessian_scaffold(
                weight,
                bad,
                body_rate_q256=512,
                input_block_size=256,
                output_block_size=1,
                input_seed=1,
                output_seed=2,
                research_opt_in=M.RESEARCH_OPT_IN,
            )

    prepared = M.prepare_one_linear_diagonal_hessian_scaffold(
        weight,
        diagonal,
        body_rate_q256=512,
        input_block_size=256,
        output_block_size=1,
        input_seed=1,
        output_seed=2,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
    forged = _rehashed_structured_prepared(
        prepared,
        lambda receipt: receipt["transformed"]["hessian_structure"].update({
            "off_block_entries_zero_by_construction": False
        }),
    )
    with pytest.raises(ValueError, match="Hessian structure mismatch"):
        M._validate_prepared_diagonal_hessian_one_linear(
            forged, body_rate_q256=512
        )
    reordered = _rehashed_structured_prepared(
        prepared,
        lambda receipt: receipt["transformed"]["hessian_structure"][
            "ordered_blocks"
        ].reverse(),
    )
    with pytest.raises(ValueError, match="Hessian structure mismatch"):
        M._validate_prepared_diagonal_hessian_one_linear(
            reordered, body_rate_q256=512
        )
