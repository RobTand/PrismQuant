from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from prismaquant.trellis_encoder import (
    TrellisEncoderError,
    encode_trellis_planes,
    snap_e2m1_scale_codes,
)
from prismaquant.trellis_formats import E2M1_FAMILY
from prismaquant.trellis_producer import encode_trellis_one_linear
from prismaquant.trellis_scale_grid import (
    SCALE_GRID_MULTIPLIERS,
    SCALE_PLANE_RATE_Q256,
    ScaleGridError,
    encode_e2m1_scale_grid_two_arm,
    propose_e2m1_scale_plane,
    require_scale_grid_selection_replay,
    require_scale_grid_source_unchanged,
    select_e2m1_scale_grid,
    validate_scale_grid_receipt,
)
from prismaquant.trellis_wire import TrellisWire, decode_values_torch


ALPHABETS = {1: (15, 11, 8, 4)}
SCHEDULE = [1] * 16 + [4] * 240
MENU = (1.0, 0.55, 0.70, 0.85, 1.15, 1.30)


def _weight(rows: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(rows, 256, generator=generator)
        * torch.exp(torch.linspace(-2.0, 2.0, 256)).reshape(1, 256)
    )


def _encoder_kwargs():
    return {
        "family": E2M1_FAMILY,
        "schedule": SCHEDULE,
        "alphabets": ALPHABETS,
        "scale_rule": "static_6",
        "sb_chunk": 2,
        "determinism_mode": "on",
        "tailbite_candidates": 4,
        "backend": "eager",
        "point_route": "full",
    }


def _grid_kwargs():
    result = _encoder_kwargs()
    result.pop("family")
    return {
        "body_rate_q256": sum(SCHEDULE),
        "layout": "tight_offsets",
        "multipliers": MENU,
        **result,
    }


def _identity(weight: torch.Tensor, metric: torch.Tensor):
    return encode_trellis_planes(weight, metric, **_encoder_kwargs())


def _scale_codes(encoded, rows: int) -> torch.Tensor:
    return torch.frombuffer(
        bytearray(encoded.scale_blob), dtype=torch.uint8
    ).reshape(rows, 16)


def _receipt_identity(receipt: dict[str, object]) -> str:
    body = {key: value for key, value in receipt.items() if key != "identity_sha256"}
    payload = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def test_candidate_menu_is_identity_plus_33_preregistered_values():
    assert len(SCALE_GRID_MULTIPLIERS) == 34
    assert SCALE_GRID_MULTIPLIERS[0] == 1.0
    assert len(set(SCALE_GRID_MULTIPLIERS)) == len(SCALE_GRID_MULTIPLIERS)


def test_proposal_identity_is_current_bytes_and_selected_cells_are_legal():
    weight = _weight()
    metric = torch.linspace(0.5, 1.5, 256)
    identity = _identity(weight, metric)
    codes = _scale_codes(identity, 2)
    proposal = propose_e2m1_scale_plane(
        weight,
        metric,
        global_scale_real=identity.global_scale_real,
        identity_scale_codes=codes,
        multipliers=MENU,
        floor_to_min_positive=True,
    )
    decoded = proposal.scale_codes.view(torch.float8_e4m3fn).float()
    assert bool(torch.isfinite(decoded).all())
    assert bool((decoded > 0).all())
    assert proposal.scale_codes.dtype == torch.uint8
    assert proposal.scale_codes.shape == codes.shape
    assert torch.all(proposal.group_sse <= proposal.identity_group_sse)


def test_shared_snap_matches_encoder_bytes_on_default_and_override_paths():
    weight = _weight()
    metric = torch.linspace(0.5, 1.5, 256)
    real_scales = weight.reshape(2, 16, 16).abs().amax(-1).clamp_min(1.0e-12) / 6.0
    default = _identity(weight, metric)
    assert torch.equal(
        default.scale_codes,
        snap_e2m1_scale_codes(
            real_scales,
            default.global_scale_real,
            multiplier=1.0,
            floor_to_min_positive=False,
        ),
    )
    frozen_global = default.global_scale_real * 1.25
    overridden = encode_trellis_planes(
        weight,
        metric,
        **_encoder_kwargs(),
        global_scale_real_override=frozen_global,
    )
    assert torch.equal(
        overridden.scale_codes,
        snap_e2m1_scale_codes(
            real_scales,
            frozen_global,
            multiplier=1.0,
            floor_to_min_positive=True,
        ),
    )


def test_proposal_refuses_identity_byte_drift():
    weight = _weight(1)
    metric = torch.ones(256)
    identity = _identity(weight, metric)
    codes = _scale_codes(identity, 1).clone()
    codes[0, 0] = codes[0, 0] + 1
    with pytest.raises(ScaleGridError, match="candidate-zero.*byte-equal"):
        propose_e2m1_scale_plane(
            weight,
            metric,
            global_scale_real=identity.global_scale_real,
            identity_scale_codes=codes,
            multipliers=MENU,
        )


@pytest.mark.parametrize("bad_code", [0x00, 0x80, 0xB8, 0x7F, 0xFF])
def test_encoder_override_refuses_nonpositive_or_nan_scale_cells(bad_code):
    weight = _weight(1)
    metric = torch.ones(256)
    identity = _identity(weight, metric)
    codes = _scale_codes(identity, 1).clone()
    codes[0, 0] = bad_code
    with pytest.raises(TrellisEncoderError, match="finite and strictly positive"):
        encode_trellis_planes(
            weight,
            metric,
            **_encoder_kwargs(),
            global_scale_real_override=identity.global_scale_real,
            scale_plane_override=codes,
        )


def test_two_full_arms_realized_gate_exact_min_and_zero_byte_rate_delta(monkeypatch):
    import prismaquant.trellis_scale_grid as module

    calls = 0
    original = module.encode_trellis_planes

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "encode_trellis_planes", counted)
    weight = _weight()
    metric = torch.ones(256)
    result = encode_e2m1_scale_grid_two_arm(
        weight, metric, **_grid_kwargs()
    )
    assert calls == 2
    assert bool(result.candidate_wins.all())
    assert torch.equal(
        result.final_tile_sse,
        torch.minimum(result.identity_tile_sse, result.candidate_tile_sse),
    )
    assert torch.all(result.final_tile_sse <= result.identity_tile_sse)
    assert len(result.wire_bytes) == len(result.identity_wire_bytes)
    assert TrellisWire.from_bytes(result.wire_bytes).to_bytes() == result.wire_bytes
    assert torch.equal(result.decoded_weight, decode_values_torch(result.wire_bytes))
    pricing = result.receipt["pricing"]
    assert pricing["scale_plane_rate_q256"] == SCALE_PLANE_RATE_Q256
    assert pricing["wire_byte_delta"] == pricing["delta_bpw_q256"] == 0
    assert result.receipt["producer_eligible"] is False


def test_identity_only_menu_runs_both_arms_and_is_byte_identical_to_old_encoder(
    monkeypatch,
):
    import prismaquant.trellis_scale_grid as module

    calls = 0
    original = module.encode_trellis_planes

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "encode_trellis_planes", counted)
    weight = _weight(1)
    metric = torch.ones(256)
    kwargs = _grid_kwargs()
    kwargs["multipliers"] = (1.0,)
    result = encode_e2m1_scale_grid_two_arm(weight, metric, **kwargs)
    legacy = encode_trellis_one_linear(
        weight,
        metric,
        family=E2M1_FAMILY,
        body_rate_q256=sum(SCHEDULE),
        schedule=SCHEDULE,
        layout="tight_offsets",
        alphabets=ALPHABETS,
        scale_rule="static_6",
        sb_chunk=2,
        determinism_mode="on",
        tailbite_candidates=4,
        backend="eager",
        point_route="full",
    )
    assert calls == 2
    assert not bool(result.candidate_wins.any())
    assert result.wire_bytes == result.identity_wire_bytes == legacy.wire_bytes
    assert result.receipt["proof"]["no_win_byte_identical"] is True


def test_realized_tie_keeps_identity_even_if_candidate_scale_bytes_differ(
    monkeypatch,
):
    import prismaquant.trellis_scale_grid as module

    original = module.propose_e2m1_scale_plane

    def different_legal_plane(*args, **kwargs):
        proposal = original(*args, **kwargs)
        changed = proposal.scale_codes.clone()
        replacement = 0x38 if int(changed[0, 0]) != 0x38 else 0x40
        changed[0, 0] = replacement
        return replace(proposal, scale_codes=changed)

    monkeypatch.setattr(module, "propose_e2m1_scale_plane", different_legal_plane)
    monkeypatch.setattr(
        module,
        "score_realized_tiles_fp64",
        lambda weight, _reconstruction, _metric: torch.zeros(
            (weight.shape[0], weight.shape[1] // 256), dtype=torch.float64
        ),
    )
    weight = _weight(1)
    kwargs = _grid_kwargs()
    kwargs["multipliers"] = (1.0, 0.75)
    result = encode_e2m1_scale_grid_two_arm(
        weight, torch.ones(256), **kwargs
    )
    assert torch.equal(result.identity_tile_sse, result.candidate_tile_sse)
    assert not bool(result.candidate_wins.any())
    assert result.wire_bytes == result.identity_wire_bytes


def test_uniformly_bad_legal_candidate_is_rejected_by_realized_gate(monkeypatch):
    import prismaquant.trellis_scale_grid as module

    original = module.propose_e2m1_scale_plane

    def bad_legal_plane(*args, **kwargs):
        proposal = original(*args, **kwargs)
        return replace(proposal, scale_codes=torch.ones_like(proposal.scale_codes))

    monkeypatch.setattr(module, "propose_e2m1_scale_plane", bad_legal_plane)
    result = encode_e2m1_scale_grid_two_arm(
        _weight(), torch.ones(256), **_grid_kwargs()
    )
    assert not bool(result.candidate_wins.any())
    assert result.wire_bytes == result.identity_wire_bytes
    assert result.receipt["proof"]["no_win_byte_identical"] is True


def test_real_shaped_c2_witness_proves_rtn_proposal_needs_realized_gate():
    generator = torch.Generator().manual_seed(14)
    weight = (
        torch.randn(2, 512, generator=generator)
        * torch.exp(torch.linspace(-3.0, 3.0, 512))[None, :]
    )
    metric = torch.rand(512, generator=generator).add_(0.01)
    kwargs = _grid_kwargs()
    kwargs["schedule"] = SCHEDULE * 2
    result = encode_e2m1_scale_grid_two_arm(weight, metric, **kwargs)

    # The RTN generator likes its joint scale plane, but the shaped trellis
    # path regresses this realized tile. The gate keeps identity here while
    # accepting strict wins elsewhere in the same multi-superblock tensor.
    assert result.identity_tile_sse[0, 0].item() == pytest.approx(
        0.20545847237462625
    )
    assert result.candidate_tile_sse[0, 0].item() == pytest.approx(
        0.2116107180942585
    )
    assert result.candidate_tile_sse[0, 0] > result.identity_tile_sse[0, 0]
    assert result.candidate_wins[0, 0].item() is False
    assert result.final_tile_sse[0, 0] == result.identity_tile_sse[0, 0]
    assert bool(result.candidate_wins.any())
    assert torch.equal(
        result.final_tile_sse,
        torch.minimum(result.identity_tile_sse, result.candidate_tile_sse),
    )


@pytest.mark.parametrize("seed", [3, 17, 91])
def test_randomized_realized_nonregression(seed):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(2, 256, generator=generator)
    metric = torch.rand(256, generator=generator).add_(0.01)
    result = encode_e2m1_scale_grid_two_arm(
        weight, metric, **_grid_kwargs()
    )
    assert torch.equal(
        result.final_tile_sse,
        torch.minimum(result.identity_tile_sse, result.candidate_tile_sse),
    )
    assert bool((result.final_tile_sse <= result.identity_tile_sse).all())


def test_splice_mutation_trips_exact_cf_minimum(monkeypatch):
    import prismaquant.trellis_scale_grid as module

    def wrong_splice(identity, _candidate, _wins):
        return identity

    monkeypatch.setattr(module, "_splice_encoded_planes", wrong_splice)
    with pytest.raises(ScaleGridError, match=r"Cf is not exactly min\(C0, C1\)"):
        encode_e2m1_scale_grid_two_arm(
            _weight(), torch.ones(256), **_grid_kwargs()
        )


def test_scorer_disagreement_cannot_certify_a_wrong_winner(monkeypatch):
    import prismaquant.trellis_scale_grid as module

    original = module.score_realized_tiles_fp64
    calls = 0

    def inconsistent(weight, reconstruction, metric):
        nonlocal calls
        calls += 1
        actual = original(weight, reconstruction, metric)
        if calls == 2:
            return torch.zeros_like(actual)
        return actual

    monkeypatch.setattr(module, "score_realized_tiles_fp64", inconsistent)
    with pytest.raises(ScaleGridError, match=r"Cf is not exactly min\(C0, C1\)"):
        encode_e2m1_scale_grid_two_arm(
            _weight(), torch.ones(256), **_grid_kwargs()
        )


def test_global_scale_drift_between_arms_is_refused(monkeypatch):
    import prismaquant.trellis_scale_grid as module

    original = module.encode_trellis_planes
    calls = 0

    def drifting(*args, **kwargs):
        nonlocal calls
        calls += 1
        encoded = original(*args, **kwargs)
        if calls == 2:
            return replace(
                encoded, global_scale_real=encoded.global_scale_real * 2.0
            )
        return encoded

    monkeypatch.setattr(module, "encode_trellis_planes", drifting)
    with pytest.raises(ScaleGridError, match="immutable global scale"):
        encode_e2m1_scale_grid_two_arm(
            _weight(), torch.ones(256), **_grid_kwargs()
        )


def test_receipt_mutation_refuses_even_when_claim_booleans_are_left_true():
    result = encode_e2m1_scale_grid_two_arm(
        _weight(1), torch.ones(256), **_grid_kwargs()
    )
    validate_scale_grid_receipt(result.receipt)
    forged = copy.deepcopy(result.receipt)
    forged["pricing"]["delta_bpw_q256"] = 1
    with pytest.raises(ScaleGridError, match="identity mismatch"):
        validate_scale_grid_receipt(forged)

    semantic_forgery = copy.deepcopy(result.receipt)
    semantic_forgery["proof"]["cf_le_c0"] = False
    semantic_forgery["identity_sha256"] = _receipt_identity(semantic_forgery)
    with pytest.raises(ScaleGridError, match="proof obligation"):
        validate_scale_grid_receipt(semantic_forgery)

    opaque_forgery = copy.deepcopy(result.receipt)
    opaque_forgery["arms"]["identity"]["wire_sha256"] = "0" * 64
    opaque_forgery["identity_sha256"] = _receipt_identity(opaque_forgery)
    # A mapping alone cannot authenticate an opaque hash. The authoritative
    # boundary reruns both arms and compares the retained artifact exactly.
    validate_scale_grid_receipt(opaque_forgery)
    forged_selection = replace(result, receipt=opaque_forgery)
    with pytest.raises(ScaleGridError, match="receipt semantics"):
        require_scale_grid_selection_replay(
            forged_selection,
            _weight(1),
            torch.ones(256),
            **_grid_kwargs(),
        )


def test_full_gate_replays_wire_receipt_and_canonical_serialization_exactly():
    kwargs = _grid_kwargs()
    first = encode_e2m1_scale_grid_two_arm(
        _weight(), torch.linspace(0.5, 1.5, 256), **kwargs
    )
    second = require_scale_grid_selection_replay(
        first,
        _weight(),
        torch.linspace(0.5, 1.5, 256),
        **kwargs,
    )
    assert first.wire_bytes == second.wire_bytes
    assert first.receipt == second.receipt
    assert TrellisWire.from_bytes(first.wire_bytes).to_bytes() == first.wire_bytes

    changed_recipe = dict(kwargs)
    changed_recipe["sb_chunk"] = 1
    with pytest.raises(ScaleGridError, match="replay recipe identity mismatch"):
        require_scale_grid_selection_replay(
            first,
            _weight(),
            torch.linspace(0.5, 1.5, 256),
            **changed_recipe,
        )


def test_render_recipe_uses_one_immutable_schedule_snapshot():
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

    schedule = FlipSchedule()
    kwargs = _grid_kwargs()
    kwargs["schedule"] = schedule
    result = encode_e2m1_scale_grid_two_arm(
        _weight(1), torch.ones(256), **kwargs
    )
    wire = TrellisWire.from_bytes(result.wire_bytes)
    assert schedule.iterations == 1
    assert list(wire.schedule) == identity_schedule
    assert result.receipt["render_recipe"]["schedule"] == identity_schedule


def test_render_recipe_refuses_string_subclass_alias_before_encode():
    class BackendAlias(str):
        def __str__(self):
            return "triton"

    kwargs = _grid_kwargs()
    kwargs["backend"] = BackendAlias("eager")
    with pytest.raises(ScaleGridError, match="backend must be a nonempty plain string"):
        encode_e2m1_scale_grid_two_arm(
            _weight(1), torch.ones(256), **kwargs
        )


def test_selector_source_drift_after_import_refuses(monkeypatch):
    import prismaquant.trellis_scale_grid as module

    monkeypatch.setattr(module, "_current_scale_grid_source_sha256", lambda: "0" * 64)
    with pytest.raises(ScaleGridError, match="source changed since module import"):
        require_scale_grid_source_unchanged()


def test_encoder_source_drift_after_scale_grid_import_refuses(monkeypatch):
    import prismaquant.trellis_encoder as encoder_module
    import prismaquant.trellis_scale_grid as module

    loaded_identity = encoder_module.encoder_source_sha256()
    monkeypatch.setattr(
        encoder_module, "_current_encoder_source_sha256", lambda: "0" * 64
    )
    assert encoder_module.encoder_source_sha256() == loaded_identity
    with pytest.raises(TrellisEncoderError, match="source changed since module import"):
        encoder_module.require_encoder_source_unchanged()
    with pytest.raises(ScaleGridError, match="encoder source changed"):
        module.require_scale_grid_encoder_source_unchanged()


def test_coupled_or_unknown_scope_and_retired_group_selector_fail_closed():
    kwargs = _grid_kwargs()
    kwargs["selection_scope"] = "row_group16"
    with pytest.raises(ScaleGridError, match="row_superblock"):
        encode_e2m1_scale_grid_two_arm(
            _weight(1), torch.ones(256), **kwargs
        )
    with pytest.raises(ScaleGridError, match="path-coupled"):
        select_e2m1_scale_grid(
            _weight(1), torch.ones(256), torch.ones(1, 16), 1.0
        )


@pytest.mark.parametrize(
    "menu, message",
    [
        ((), "candidate zero"),
        ((0.75, 1.0), "candidate zero"),
        ((1.0, 1.0), "unique"),
        ((1.0, 0.0), "finite and positive"),
        ((1.0, float("nan")), "finite and positive"),
        ((1.0, float("inf")), "finite and positive"),
        ((True, 0.75), "plain numeric"),
        ((1.0, "0.75"), "plain numeric"),
    ],
)
def test_malformed_candidate_menu_refuses(menu, message):
    kwargs = _grid_kwargs()
    kwargs["multipliers"] = menu
    with pytest.raises(ScaleGridError, match=message):
        encode_e2m1_scale_grid_two_arm(
            _weight(1), torch.ones(256), **kwargs
        )
