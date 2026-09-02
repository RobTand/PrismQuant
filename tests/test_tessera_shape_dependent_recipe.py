"""The seam under a wire recipe whose size is a function of the shape.

Tessera's E4M3 grid is scheduled to move from the TCQ body over a block scale
plane to the **window body over the CHANNEL plane**
(``tessera.export.wire_recipe``'s own docstring names the flip and its gates;
measured 0.985x of EXL3 K4 at 4.0 bpp).  That recipe charges two things no
bits-per-parameter rate can state: one fp16 per output row, and one
``2**L``-byte window table per unit.  The same rung is 4.0078 bpp on a
2048x4096 tensor and 4.4653 bpp on a 96x768 one.

These tests run the seam in that world by substituting ``wire_recipe`` for the
E4M3 grids only, and they assert the substitution took effect before asserting
anything else -- the recipe lookup is memoised, so a test that forgot to clear
the cache would otherwise pass while measuring today's wire.

The authority for every number here is ``tessera.calculator.terminal_rate``,
called directly with the plane flags the recipe implies, never through the code
under test.
"""

from fractions import Fraction
import math
import pickle

import pytest

import prismaquant.footprint as fp
import prismaquant.format_registry as fr
import prismaquant.tessera_formats as tfm
from prismaquant.tessera_allocator import build_tessera_allocator_candidate
from prismaquant.tessera_footprint import (
    TesseraShapeRate,
    tessera_exact_bits_for_shape,
)

tessera_export = pytest.importorskip("tessera.export")
from tessera.calculator import terminal_rate  # noqa: E402
from tessera.export import WireRecipe  # noqa: E402
from tessera.manifest import BodyKind, ScalePlaneKind  # noqa: E402

WINDOW_BITS = 12
#: The recipe Tessera ships for E4M3 as of 2026-09-02, by reference rather
#: than by value: this file must keep testing whatever that recipe becomes.
LIVE_RECIPE = tessera_export.E4M3_RECIPE
LIVE_WINDOW_BITS = tessera_export.E4M3_WINDOW_BITS

#: A second window width, substituted for E4M3 by the fixture below.  Not a
#: hypothetical any more -- the flip has landed -- but a different L keeps the
#: tests honest about pricing the recipe rather than a memorised constant.
FLIPPED = WireRecipe(
    body=BodyKind.WINDOW,
    span=1,
    scale_plane=ScalePlaneKind.CHANNEL,
    window_bits=WINDOW_BITS,
    window_seed=tessera_export.DEFAULT_WINDOW_SEED,
    window_sigma=tessera_export.DEFAULT_WINDOW_SIGMA,
    channel_sigma=None,
)
SHAPES = ((2048, 4096), (96, 768))
RUNGS = (512, 1024, 2048)


def _reference_bits(
    rung: int, rows: int, columns: int, window_bits: int = WINDOW_BITS,
) -> Fraction:
    """Tessera's own accountant, told the recipe by hand."""

    rate = terminal_rate(
        rung,
        rows,
        columns,
        with_scale_base=False,   # CHANNEL has no block plane at all
        with_scale_refine=False,
        with_row_scale=True,     # one fp16 per output row on DIAG_SV
        with_diagonals=False,
        completion=0,            # the window body has no completion axis
        cap=8,                   # payload_bits, not payload_bits - 1
        arity=1,
        span=1,
        window_bits=window_bits,
    )
    return rate * rows * columns


@pytest.fixture()
def flipped_e4m3(monkeypatch):
    """Serve the window/CHANNEL recipe for E4M3 grids and nothing else."""

    real = tessera_export.wire_recipe
    e4m3 = {id(tfm._build_grid("E4M3", 8, k)) for k in (1, 2)}

    def wire_recipe(grid, q256=None):
        return FLIPPED if id(grid) in e4m3 else real(grid, q256)

    monkeypatch.setattr(tessera_export, "wire_recipe", wire_recipe)
    tfm.clear_recipe_cache()
    # Prove the substitution is visible through the seam BEFORE any test body
    # relies on it: `_recipe_for` is memoised, and a stale hit would make every
    # assertion below a statement about today's wire wearing tomorrow's name.
    assert tfm.tessera_wire_recipe("TESSERA_E4M3_K1", 1024) == FLIPPED
    assert tfm.tessera_wire_recipe("TESSERA_E2M1_K2", 896).body is BodyKind.TCQ
    yield
    tfm.clear_recipe_cache()


def test_the_flip_makes_the_rung_synthesizable_not_unsynthesizable(flipped_e4m3):
    """The whole point: an E4M3 rung still resolves, and it resolves to W8A8.

    Before this change the seam refused a per-unit recipe in the shape-free
    accountant -- correct, and it would have made every E4M3 rung
    unsynthesizable on the day Tessera flipped.
    """

    spec = fr.get_format("TESSERA_E4M3_K1_R1024")
    assert spec.name == "TESSERA_E4M3_K1_R1024"
    # The decoded tile is a stock per-channel FP8 tensor, so the route is the
    # FP8 row's, by reference.
    fp8 = fr.get_format("FP8_E4M3")
    assert (spec.act_bits, spec.act_dtype_name) == (8, "fp8_e4m3")
    assert spec.act_group_size == 0
    assert spec.min_capability_sm == fp8.min_capability_sm
    assert spec.act_quant_changes_input
    assert (
        spec.activation_quantize_dequantize
        is fp8.activation_quantize_dequantize
    )
    assert spec.group_size == 0          # per output channel, FP8's spelling
    assert spec.scale_bits == 16
    assert spec.weight_element_dtype == "tessera_e4m3_k1"
    # The serving lane is still absent, and this change does not invent one.
    assert spec.producer_eligible is False


def test_the_price_is_a_function_and_it_is_terminal_rate(flipped_e4m3):
    """``bits_for_shape`` equals Tessera's accountant exactly, at every shape."""

    for rung in RUNGS:
        spec = fr.get_format(f"TESSERA_E4M3_K1_R{rung}")
        for rows, columns in SHAPES:
            want = _reference_bits(rung, rows, columns)
            assert spec.bits_for_shape((rows, columns)) == want
            assert spec.memory_bytes_for_shape((rows, columns)) == math.ceil(
                want / 8
            )
            # The same number, asked of the footprint module directly.
            assert tessera_exact_bits_for_shape(
                "TESSERA_E4M3_K1", rung, (rows, columns)
            ) == want


def test_the_rate_really_does_depend_on_the_shape(flipped_e4m3):
    """Without this, the tests above would pass under a shape-free wire too."""

    spec = fr.get_format("TESSERA_E4M3_K1_R1024")
    wide = spec.effective_bits_for_shape((2048, 4096))
    narrow = spec.effective_bits_for_shape((96, 768))
    assert wide == pytest.approx(4.0078125)
    assert narrow == pytest.approx(4.465277777777778)
    # A single scalar rate would have priced the narrow tensor 0.457 bpp light,
    # which is the whole reason this axis exists.
    assert narrow - wide > 0.4


def test_the_shape_free_rate_is_refused_never_floored(flipped_e4m3):
    """No scalar exists, so the spec has none and the property says so."""

    spec = fr.get_format("TESSERA_E4M3_K1_R1024")
    assert spec.exact_bits_per_param is None
    assert spec.bits_for_shape_fn is not None
    with pytest.raises(ValueError, match="shape-dependent"):
        spec.effective_bits


def test_a_rank_that_names_no_units_is_refused_rather_than_guessed(flipped_e4m3):
    """A per-unit plane needs to know how many units there are.

    2-D is one unit and 3-D is ``experts`` of them (below); every other rank
    says nothing about unit count, so it is refused -- as a ``ValueError``,
    which is what every accountant in the tree already catches as "this format
    cannot take this tensor" (``allocator._sort_specs_by_serialized_rate``).
    """

    spec = fr.get_format("TESSERA_E4M3_K1_R1024")
    with pytest.raises(ValueError):
        spec.memory_bytes_for_shape((768,))
    with pytest.raises(ValueError):
        spec.bits_for_shape((2, 8, 96, 768))


def test_an_allocator_candidate_is_priced_at_its_own_tensor(flipped_e4m3):
    """Two shapes, one rung, two prices -- each its tensor's."""

    family = tfm.get_tessera_family("TESSERA_E4M3_K1")
    seen = []
    for rows, columns in SHAPES:
        candidate = build_tessera_allocator_candidate(
            "model.layers.0.mlp.down_proj",
            (rows, columns),
            family=family,
            body_rate_q256=1024,
            layout="tight",
            schedule=family.column_schedule(1024, columns),
            alphabets={},
            predicted_dloss=1e-3,
        )
        want = _reference_bits(1024, rows, columns)
        assert candidate.memory_bytes == math.ceil(want / 8)
        assert candidate.footprint["body_kind"] == "window"
        assert candidate.footprint["window_bits"] == WINDOW_BITS
        assert candidate.footprint["scale_contract"] == "channel"
        assert candidate.footprint["activation_contract"] == (
            "w8a8-dynamic-e4m3-channel"
        )
        seen.append(candidate.bits_per_param)
    assert seen[1] > seen[0]


def test_an_assignment_footprint_prices_the_unit_at_its_shape(flipped_e4m3):
    """The model-level accountant inherits the fix through one funnel.

    ``footprint.assignment_artifact_bytes`` reaches every format through
    ``memory_bytes_for_shape``, so no per-consumer change was needed -- and
    that is the property worth pinning, because the day it stops being true is
    the day a Tessera unit is priced by a rate again.
    """

    rows, columns = 96, 768
    stats = {
        "layer.w": {
            "n_params": rows * columns,
            "in_features": columns,
            "out_features": rows,
        }
    }
    report = fp.assignment_artifact_bytes(
        {"layer.w": "TESSERA_E4M3_K1_R1024"},
        stats,
        source_total_bytes=rows * columns * 2 + 1600,
        regime="bf16",
        source_manifest=None,
    )
    want = math.ceil(_reference_bits(1024, rows, columns) / 8)
    assert report["body_quant_bytes"] == want
    assert report["artifact_bytes"] == 1600 + want


def test_the_shape_rate_is_comparable_and_picklable(flipped_e4m3):
    """``get_format`` builds a fresh spec every call; a lambda would not do."""

    first = fr.get_format("TESSERA_E4M3_K1_R1024").bits_for_shape_fn
    second = fr.get_format("TESSERA_E4M3_K1_R1024").bits_for_shape_fn
    assert first == second
    assert isinstance(first, TesseraShapeRate)
    assert pickle.loads(pickle.dumps(first))((96, 768)) == first((96, 768))
    # A different rung is a different price.
    other = fr.get_format("TESSERA_E4M3_K1_R2048").bits_for_shape_fn
    assert other != first


def test_the_e2m1_families_do_not_move_when_e4m3_flips(flipped_e4m3):
    """The flip is per grid, and the seam must not generalise it."""

    spec = fr.get_format("TESSERA_E2M1_K2_R896")
    assert spec.bits_for_shape_fn is None
    assert spec.exact_bits_per_param == Fraction(4)
    # ``effective_bits`` is the registry's legacy weight-plus-group-scale
    # model, untouched here; what matters is that it answers rather than
    # refusing, because this rung HAS a rate.
    assert spec.effective_bits == pytest.approx(4.25)
    assert spec.bits_for_shape((2048, 4096)) == Fraction(4) * 2048 * 4096
    family = tfm.get_tessera_family("TESSERA_E2M1_K2")
    assert tfm.family_rate_cap(family) == 7
    assert tfm.family_q256_bounds(family) == (128, 896)
    # ... while E4M3's bounds widen to the window body's cap.
    e4m3 = tfm.get_tessera_family("TESSERA_E4M3_K1")
    assert tfm.family_rate_cap(e4m3) == 8
    assert tfm.family_q256_bounds(e4m3) == (256, 2048)


def test_the_unpatched_wire_for_e4m3_is_the_window_over_channel():
    """No fixture: the shipping recipe, asserted against tessera's own object.

    The flip landed on 2026-09-02, so the world these tests were written for is
    now the default.  This is stated by equality with
    ``tessera.export.E4M3_RECIPE`` rather than by repeating its fields, so a
    later change to L or to the plane arrives here as a changed price rather
    than as a passing test about a recipe nobody writes.
    """

    family = tfm.get_tessera_family("TESSERA_E4M3_K1")
    for rung in (256, 1024, 2048):
        assert tfm.tessera_wire_recipe(family, rung) == LIVE_RECIPE
    assert LIVE_RECIPE.body is BodyKind.WINDOW
    assert LIVE_RECIPE.scale_plane is ScalePlaneKind.CHANNEL
    assert not tfm.recipe_is_shape_free(LIVE_RECIPE)

    # Priced by a function, not by a rate, and exact against terminal_rate at
    # the width tessera actually ships.
    spec = fr.get_format("TESSERA_E4M3_K1_R1024")
    assert spec.exact_bits_per_param is None
    assert spec.bits_for_shape_fn is not None
    for shape in SHAPES:
        assert spec.bits_for_shape(shape) == _reference_bits(
            1024, *shape, window_bits=LIVE_WINDOW_BITS)
    with pytest.raises(ValueError, match="shape-dependent"):
        spec.effective_bits

    # The window body's ceiling, and the W8A8 route, with nothing patched.
    assert tfm.family_rate_cap(family) == 8
    assert tfm.family_q256_bounds(family) == (256, 2048)
    fp8 = fr.get_format("FP8_E4M3")
    assert (spec.act_bits, spec.act_dtype_name, spec.act_group_size) == (
        8, "fp8_e4m3", 0)
    assert spec.min_capability_sm == fp8.min_capability_sm
    assert spec.producer_eligible is False


def test_the_sub_cap_e2m1x2_wire_is_shape_dependent_too():
    """The other half of the flip: E2M1x2 below the coset trellis's cap.

    ``wire_recipe`` returns the window body over LUT16 there (L=12) and the
    trellis at the cap, so one family carries both kinds of price and the seam
    has to answer per rung rather than per family.
    """

    from tessera.export import (
        E2M1X2_SUBCAP_RECIPE, TCQ_RECIPE, tcq_cap_q256,
    )

    family = tfm.get_tessera_family("TESSERA_E2M1_K2")
    cap = tcq_cap_q256(family.payload_grid())
    assert cap == 896
    assert tfm.tessera_wire_recipe(family, cap - 1) == E2M1X2_SUBCAP_RECIPE
    assert tfm.tessera_wire_recipe(family, cap) == TCQ_RECIPE

    sub_cap = fr.get_format("TESSERA_E2M1_K2_R768")
    at_cap = fr.get_format("TESSERA_E2M1_K2_R896")
    # Below the cap: a per-unit table, so a function and no scalar rate.
    assert sub_cap.exact_bits_per_param is None
    assert sub_cap.bits_for_shape_fn is not None
    with pytest.raises(ValueError, match="shape-dependent"):
        sub_cap.effective_bits
    # At the cap: the coset trellis, a rate, and still exactly 4.0 bpp -- the
    # rung every published Tessera number is quoted at.
    assert at_cap.exact_bits_per_param == Fraction(4)
    assert at_cap.bits_for_shape_fn is None


# ---------------------------------------------------------------------------
# Packed MoE experts: one encoded unit per expert
# ---------------------------------------------------------------------------
# A routed-expert stack arrives at the accountant as ``(experts, out, in)``
# (``allocator_solver._shape_from_stats``), and under a per-unit recipe the
# price depends on how many units that is.  It is E of them: the exporter
# encodes per source tensor name, the trellis runs down rows within a column so
# a fused stack could not be decoded one expert at a time, and the kernel lane
# decodes each unit against its own window table.  These tests pin that, and
# pin that the two live consumers which used to raise now reach a number.
#
# They run **unpatched**, on the shipping E4M3 recipe: routed experts are what
# this format exists for, so pricing them is a claim about the wire tessera
# writes today rather than about a substituted one.

GLM_EXPERTS = 128
GLM_EXPERT_SHAPE = (1408, 4096)
GLM_PACKED_SHAPE = (GLM_EXPERTS, *GLM_EXPERT_SHAPE)


def test_a_packed_expert_stack_is_priced_as_one_unit_per_expert():
    spec = fr.get_format("TESSERA_E4M3_K1_R1024")
    per_expert_bits = _reference_bits(
        1024, *GLM_EXPERT_SHAPE, window_bits=LIVE_WINDOW_BITS)
    per_expert_bytes = math.ceil(per_expert_bits / 8)

    assert spec.bits_for_shape(GLM_PACKED_SHAPE) == GLM_EXPERTS * per_expert_bits
    assert spec.memory_bytes_for_shape(GLM_PACKED_SHAPE) == (
        GLM_EXPERTS * per_expert_bytes
    )
    # The fused reading -- one unit of (experts*out, in) -- would charge one
    # window table instead of 128.  Naming the difference is what makes this a
    # test of the convention rather than of the arithmetic.
    fused_bits = _reference_bits(
        1024, GLM_EXPERTS * GLM_EXPERT_SHAPE[0], GLM_EXPERT_SHAPE[1],
        window_bits=LIVE_WINDOW_BITS)
    assert spec.bits_for_shape(GLM_PACKED_SHAPE) - fused_bits == (
        (GLM_EXPERTS - 1) * (1 << LIVE_WINDOW_BITS) * 8
    )
    # 1-D and 4-D remain refused: neither says how many units it is.
    with pytest.raises(ValueError):
        spec.bits_for_shape((4096,))
    with pytest.raises(ValueError):
        spec.bits_for_shape((2, 128, 1408, 4096))


def test_the_stats_shape_a_packed_expert_really_produces():
    """The shape under test is the one the pipeline builds, not one invented."""

    from prismaquant.allocator_solver import _shape_from_stats

    entry = {
        "num_experts": GLM_EXPERTS,
        "out_features": GLM_EXPERT_SHAPE[0],
        "in_features": GLM_EXPERT_SHAPE[1],
        "n_params": GLM_EXPERTS * GLM_EXPERT_SHAPE[0] * GLM_EXPERT_SHAPE[1],
    }
    assert _shape_from_stats(entry) == GLM_PACKED_SHAPE
    spec = fr.get_format("TESSERA_E4M3_K1_R1024")
    assert spec.memory_bytes_for_shape(_shape_from_stats(entry)) > 0


def test_the_allocator_reaches_a_rate_for_a_packed_expert_format():
    """``_sort_specs_by_serialized_rate`` used to skip the tensor, then raise.

    Skipping every expert tensor left ``total_params == 0``, which fell through
    to ``float(spec.effective_bits)`` -- and that property refuses for a
    shape-dependent recipe.  The rate pass would have died on exactly the
    tensors Tessera is for.
    """

    from prismaquant.allocator import _sort_specs_by_serialized_rate

    spec = fr.get_format("TESSERA_E4M3_K1_R1024")
    stats = {
        "model.layers.0.mlp.experts.down_proj": {
            "num_experts": GLM_EXPERTS,
            "out_features": GLM_EXPERT_SHAPE[0],
            "in_features": GLM_EXPERT_SHAPE[1],
            "n_params": GLM_EXPERTS * GLM_EXPERT_SHAPE[0] * GLM_EXPERT_SHAPE[1],
        }
    }
    _ordered, rates = _sort_specs_by_serialized_rate([spec], stats, None)
    want = _reference_bits(
        1024, *GLM_EXPERT_SHAPE, window_bits=LIVE_WINDOW_BITS
    ) / (GLM_EXPERT_SHAPE[0] * GLM_EXPERT_SHAPE[1])
    assert rates[spec.name] == pytest.approx(float(want))


def test_decision_unit_construction_survives_a_packed_expert():
    """The unpriced path: ``decision_units`` calls the accountant with no try.

    ``target_profile="research"`` because the production packed-MoE profile
    denies every Tessera rung for expert tensors (no serving lane -- principle
    9), so that path never reaches the accountant at all today.  The research
    profile is the one the Tessera allocator itself prices under, and it is
    where a 3-D refusal would have surfaced as a crash rather than a skip.
    """

    import torch
    import torch.nn as nn

    from prismaquant.decision_units import discover_units
    from prismaquant.model_profiles.qwen3 import Qwen3Profile

    experts, out_features, in_features = 4, 512, 256

    class _PackedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(
                torch.zeros(experts, 2 * out_features, in_features))
            self.down_proj = nn.Parameter(
                torch.zeros(experts, in_features, out_features))

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.experts = _PackedExperts()

    class _Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([_Layer()])

    spec = fr.get_format("TESSERA_E4M3_K1_R1024")
    blocks, singletons, n_params = discover_units(
        _Toy(), Qwen3Profile(), [spec, fr.get_format("BF16")],
        target_profile="research",
    )
    units = [u for group in blocks.values() for u in group] + list(singletons)
    assert units
    priced = [
        option
        for unit in units
        for option in unit.options
        if option.fmt == spec.name
    ]
    assert priced, "the Tessera rung was dropped from every unit's menu"
    for option in priced:
        # Each member is a packed (experts, out, in) stack, and its price is
        # the per-expert figure times the expert count -- reached through
        # ``memory_bytes_for_shape`` with no try/except in between.
        assert option.memory_bytes > 0
    want = sum(
        experts * math.ceil(
            _reference_bits(1024, shape[0], shape[1],
                            window_bits=LIVE_WINDOW_BITS) / 8
        )
        for shape in ((2 * out_features, in_features), (in_features, out_features))
    )
    assert sum(option.memory_bytes for option in priced) == want
