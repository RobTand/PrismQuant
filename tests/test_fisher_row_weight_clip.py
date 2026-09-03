"""Single-source guard for the Fisher row-weight clip (issue #159).

The mean-1 normalise-then-clip rule used to be copied across the render
scorer, the quant-cost probe, and the native exporter, each with its own env
read, so the cost paths could silently disagree about the same quantity. The
rule now lives in ``prismaquant.render_score`` (see
``normalize_clipped_fisher_row_weights`` and
``resolve_fisher_row_weight_clip``). These tests pin that: every path must
resolve the same clip from the same env vars and produce identical weights.
"""

from __future__ import annotations

import pytest
import torch

CANONICAL_ENV = "PRISMAQUANT_FISHER_OUTPUT_MSE_ROW_WEIGHT_CLIP"
ALIAS_ENV = "PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP"
_CPU = torch.device("cpu")


def _clear_clip_env(monkeypatch) -> None:
    monkeypatch.delenv(CANONICAL_ENV, raising=False)
    monkeypatch.delenv(ALIAS_ENV, raising=False)


def _render_path(row_weights: torch.Tensor, n_rows: int) -> torch.Tensor | None:
    from prismaquant.render_score import normalize_row_weights

    return normalize_row_weights(row_weights, n_rows, _CPU)


def _cost_path(row_weights: torch.Tensor, n_rows: int) -> torch.Tensor | None:
    from prismaquant.measure_quant_cost import (
        _normalize_fisher_output_mse_row_weights,
    )

    return _normalize_fisher_output_mse_row_weights(
        row_weights, torch.arange(n_rows), n_rows, _CPU
    )


def _export_path(row_weights: torch.Tensor, n_rows: int) -> torch.Tensor | None:
    from prismaquant.export_native_compressed import _normalize_fisher_row_weights

    return _normalize_fisher_row_weights(row_weights, n_rows, _CPU)


def _all_paths(
    row_weights: torch.Tensor, n_rows: int
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    return (
        _render_path(row_weights, n_rows),
        _cost_path(row_weights, n_rows),
        _export_path(row_weights, n_rows),
    )


def _dense_vector() -> torch.Tensor:
    # Mean-1 normalises to [2.4, 0.8 * 7]: a clip of 2.0 binds on the hot
    # element without the post-clip renormalisation hiding it.
    return torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def test_canonical_env_drives_all_paths(monkeypatch) -> None:
    """One env var, one answer on every path.

    Pre-fix failure: with only the canonical var set, the cost path clipped
    (hot element ~2.105) while the scorer and exporter read the alias only
    and returned the unclipped 2.4.
    """
    _clear_clip_env(monkeypatch)
    monkeypatch.setenv(CANONICAL_ENV, "2.0")
    render_out, cost_out, export_out = _all_paths(_dense_vector(), 8)
    assert render_out is not None and cost_out is not None
    assert export_out is not None
    assert torch.equal(render_out, cost_out)
    assert torch.equal(render_out, export_out)
    # The clip binds: without it the hot element would be 2.4.
    assert float(render_out[0]) < 2.4


def test_alias_selects_same_value_as_canonical(monkeypatch) -> None:
    """The old env var name keeps working as a documented alias."""
    vec = _dense_vector()
    _clear_clip_env(monkeypatch)
    monkeypatch.setenv(ALIAS_ENV, "2.0")
    via_alias = _all_paths(vec, 8)
    _clear_clip_env(monkeypatch)
    monkeypatch.setenv(CANONICAL_ENV, "2.0")
    via_canonical = _all_paths(vec, 8)
    for alias_out, canonical_out in zip(via_alias, via_canonical):
        assert alias_out is not None and canonical_out is not None
        assert torch.equal(alias_out, canonical_out)
    # And the alias on its own also clips (not silently ignored).
    assert via_alias[0] is not None and float(via_alias[0][0]) < 2.4


def test_canonical_wins_when_both_set(monkeypatch) -> None:
    """Precedence matches the historical cost-probe rule: canonical first."""
    _clear_clip_env(monkeypatch)
    monkeypatch.setenv(CANONICAL_ENV, "2.0")
    monkeypatch.setenv(ALIAS_ENV, "33.0")
    render_out, cost_out, export_out = _all_paths(_dense_vector(), 8)
    assert render_out is not None and cost_out is not None
    assert export_out is not None
    assert torch.equal(render_out, cost_out)
    assert torch.equal(render_out, export_out)
    assert float(render_out[0]) < 2.4


def test_paths_route_through_shared_helper(monkeypatch) -> None:
    """Every path must call the single-sourced helper, not its own copy.

    If someone reintroduces a second copy of the normalise-then-clamp
    arithmetic in any wrapper, that wrapper stops calling the helper and
    this test fails.
    """
    import prismaquant.export_native_compressed as export_mod
    import prismaquant.measure_quant_cost as cost_mod
    import prismaquant.render_score as score_mod

    real = score_mod.normalize_clipped_fisher_row_weights
    calls: list[tuple[float, bool]] = []

    def spy(
        rw: torch.Tensor, clip: float, *, require_positive_mean: bool
    ) -> torch.Tensor | None:
        calls.append((float(clip), bool(require_positive_mean)))
        return real(rw, clip, require_positive_mean=require_positive_mean)

    monkeypatch.setattr(
        score_mod, "normalize_clipped_fisher_row_weights", spy
    )
    monkeypatch.setattr(
        cost_mod, "normalize_clipped_fisher_row_weights", spy
    )
    monkeypatch.setattr(
        export_mod, "normalize_clipped_fisher_row_weights", spy
    )
    _clear_clip_env(monkeypatch)
    render_out, cost_out, export_out = _all_paths(_dense_vector(), 8)
    assert render_out is not None and cost_out is not None
    assert export_out is not None
    # One helper call per path: the scorer keeps the legacy lenient policy
    # while the probe and exporter keep the strict positive-mean policy.
    assert calls == [(64.0, False), (64.0, True), (64.0, True)]
    # The spy delegated, so outputs are the real clipped weights.
    assert torch.equal(render_out, cost_out)
    assert torch.equal(render_out, export_out)


def test_default_weights_are_bit_identical(monkeypatch) -> None:
    """Numeric identity: hand-computed exact outputs, default clip.

    Both vectors use exact binary arithmetic, so torch.equal is exact, not
    approximate. These pass identically before and after the fix; they prove
    the single-sourcing changed nothing numerically.
    """
    _clear_clip_env(monkeypatch)
    # Fully concentrated row over 128 rows: mean-1 value is exactly n_rows,
    # the clip binds, and the post-clip renormalisation restores it exactly.
    hot = torch.zeros(128)
    hot[0] = 128.0
    expected_hot = torch.zeros(128)
    expected_hot[0] = 128.0
    for out in _all_paths(hot, 128):
        assert out is not None
        assert torch.equal(out, expected_hot)
    # Dense vector below the clip: pure mean-1 normalisation.
    dense = torch.tensor([4.0, 2.0, 2.0, 0.0])
    expected_dense = torch.tensor([2.0, 1.0, 1.0, 0.0])
    for out in _all_paths(dense, 4):
        assert out is not None
        assert torch.equal(out, expected_dense)


def test_binding_clip_is_exact(monkeypatch) -> None:
    """Pin the clip-then-renormalise arithmetic with exact binary values."""
    _clear_clip_env(monkeypatch)
    monkeypatch.setenv(CANONICAL_ENV, "1.0")
    # Mean-1 gives [2, 0]; the clip binds to [1, 0]; renormalisation by the
    # post-clip mean 0.5 restores [2, 0]. Every step is exact in float32.
    vec = torch.tensor([2.0, 0.0])
    expected = torch.tensor([2.0, 0.0])
    for out in _all_paths(vec, 2):
        assert out is not None
        assert torch.equal(out, expected)


def test_resolver_defaults_and_fallbacks(monkeypatch) -> None:
    from prismaquant.render_score import resolve_fisher_row_weight_clip

    _clear_clip_env(monkeypatch)
    assert resolve_fisher_row_weight_clip() == pytest.approx(64.0)
    monkeypatch.setenv(ALIAS_ENV, "7.5")
    assert resolve_fisher_row_weight_clip() == pytest.approx(7.5)
    monkeypatch.setenv(CANONICAL_ENV, "3.25")
    assert resolve_fisher_row_weight_clip() == pytest.approx(3.25)
    monkeypatch.setenv(CANONICAL_ENV, "not-a-number")
    assert resolve_fisher_row_weight_clip() == pytest.approx(64.0)
