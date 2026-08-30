from __future__ import annotations

import copy
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
    prepared = M.prepare_one_linear_scaffold(
        weight,
        torch.eye(512),
        body_rate_q256=512,
        input_block_size=16,
        output_block_size=1,
        input_seed=1,
        output_seed=2,
        research_opt_in=M.RESEARCH_OPT_IN,
    )
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
    assert receipt["block_ldl"]["terminal_dense_D_consumed"] is False
    assert receipt["block_ldl"][
        "terminal_metric_mode"
    ] == terminal_metric_mode
    assert receipt["block_ldl"]["block_size"] == 256
    assert receipt["block_ldl"]["buffer_blocks"] == buffer_blocks
    assert receipt["same_byte_reparse_verified"] is True
    assert receipt["serve_algebra"]["wire_identity_verified"] is True
    assert receipt["producer_eligible"] is False
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
