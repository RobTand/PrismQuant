"""A narrowed production-cache render must keep the unstriped run's rows.

Rows feed the GPTQ Hessian; the Hessian feeds the rendered bytes. So a build
that narrows which Linears the collector hooks renders different bytes than a
full build of the same recipe -- two artifacts wearing one name, which is
exactly what principle 8 (one rendering behind the surrogate, the KL and the
exported bytes) exists to prevent.

``56c765d`` closed the ``--resume`` half by drawing reservoir priorities for
every *hooked* Linear. Two narrowings reached the **hook** set itself and were
out of its reach:

  * ``build_production_cache.py`` -- ``--include-qnames-file`` shortened the
    ``qnames`` list before the fill call, i.e. every
    ``production_cache_stripes`` stripe (#130);
  * ``production_weight_cache.py`` -- ``qname_set`` was the
    assignment-narrowed render set AND the collector's hook set, so every
    ``--render-scope assignment`` build (``run-pipeline.sh:590``, the shipping
    default) hooked fewer Linears than a ``format-menu`` build of the same
    model (#135).

Both are closed here. The collector hooks the caller's whole
``eligible_qnames`` enumeration and narrowing arrives as ``render_assignment``
or ``render_qnames``, which gives the invariant these tests hold to:

    the bytes of a (qname, fmt) pair are a function of ``qnames`` and the
    calibration, never of which subset a call renders.

One narrowing is deliberately still byte-moving and is pinned as such:
shortening ``qnames`` itself. That is now a caller error the docstring names,
not the way a stripe is expressed --
``test_shortening_the_enumeration_still_moves_the_bytes`` exists so nobody
re-learns it on a 90 GB cache.

The full digest matrix, the N_CALIB discriminator (bin shape vs shared stream),
the real-assignment census and the machine-readable receipt live in
``experiments/stripe_row_identity_byte_baseline.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_EXPERIMENT = REPO_ROOT / "experiments" / "stripe_row_identity_byte_baseline.py"


@pytest.fixture(scope="module")
def bb():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_pq130_bb", _EXPERIMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frozen(bb):
    """One frozen model + calibration + reference render, shared by every arm."""
    state = bb.frozen_state()
    calib = bb.frozen_calib()
    reference = bb.run_arm(state, calib, bb.all_qnames())
    return state, calib, reference


def test_the_reservoir_is_actually_selecting(bb, frozen):
    """Without selection pressure every arm agrees for an unrelated reason."""
    _, _, reference = frozen
    kept = sorted({int(shape[0]) for shape in reference["row_shapes"].values()})
    assert kept == [bb.MAX_ACT_ROWS], (
        f"the reservoir kept {kept} of {bb.N_CALIB * bb.SEQLEN} candidate rows; "
        "with no selection pressure a row-identity comparison proves nothing"
    )


def test_the_render_is_deterministic(bb, frozen):
    """Without determinism no arm comparison is readable."""
    state, calib, reference = frozen
    control = bb.run_arm(state, calib, bb.all_qnames())
    assert control["weights"] == reference["weights"]


def test_assignment_render_scope_keeps_the_format_menu_rows(bb, frozen):
    """#135: the shipping default's scope switch must not move the bytes.

    Same ``qnames`` argument in both arms. The only difference is whether the
    narrowing arrives as ``render_assignment``. Under the shipping default
    ``SELECTION_MODE=surrogate`` the AURA dW cost cache is built
    ``format-menu`` and the export cache ``assignment``, so this equality is
    what makes the surrogate's rendered dW the bytes that ship.
    """
    state, calib, reference = frozen
    stripe = set(bb.layer_qnames((1, 3)))
    assigned = {
        q: (bb.FORMAT if q in stripe else "BF16") for q in bb.all_qnames()
    }
    arm = bb.run_arm(
        state, calib, bb.all_qnames(), render_assignment=assigned
    )
    rows, byts = bb.compare(arm, reference)
    assert len(arm["weights"]) == len(stripe)
    assert not rows, (
        f"--render-scope assignment kept different rows on {len(rows)} units "
        f"than --render-scope format-menu. Re-run {_EXPERIMENT.name}."
    )
    assert not byts, (
        f"--render-scope assignment rendered {len(byts)} units differently "
        "from --render-scope format-menu: the cost cache and the export cache "
        "are two renderings again (#135)."
    )


@pytest.mark.parametrize(
    "label,indices",
    [
        ("non-prefix stripe (the plan_stripes LPT bin shape)", (1, 3)),
        ("prefix stripe (a contiguous partition)", (0, 1)),
    ],
)
def test_a_stripe_expressed_as_render_qnames_keeps_the_unstriped_bytes(
    bb, frozen, label, indices,
):
    """#130: a stripe must reproduce the unsharded run's bytes exactly.

    Both bin shapes, because the issue blamed ``plan_stripes``'s LPT binning.
    It was never the binning -- see the N_CALIB sweep in the harness -- so a
    contiguous partition must pass here for the same reason a non-prefix one
    does, and neither may be the thing that makes it pass.
    """
    state, calib, reference = frozen
    arm = bb.run_arm(
        state, calib, bb.all_qnames(), render_qnames=bb.layer_qnames(indices)
    )
    rows, byts = bb.compare(arm, reference)
    assert len(arm["weights"]) == 4, "arm did not render the expected units"
    assert not rows, f"{label}: {len(rows)} units kept different rows"
    assert not byts, (
        f"{label}: {len(byts)} units rendered different bytes than the "
        f"unstriped run. Re-run {_EXPERIMENT.name} before editing this test."
    )


def test_one_bf16_unit_last_in_the_enumeration_does_not_move_the_bytes(
    bb, frozen,
):
    """The real assignment shape, not a half-the-model carve.

    A production ``layer_config.json`` BF16s a handful of units: 4 of 504 on
    ``qwen38-27b-scout20``, 228 of 614 on ``prod-27b-nvfp4cb-5p5``. The
    hardest case for "surely four units cannot matter" is ONE BF16 unit last
    in the enumeration -- within a single forward pass every earlier hook
    fires at the same stream offset, so a one-sample run would agree anyway.
    It is the second calibration sample that re-enters the stream behind a
    shorter hook set, and production runs 32.
    """
    state, calib, reference = frozen
    full = bb.all_qnames()
    assert bb.N_CALIB >= 2, "this case only bites from the 2nd sample on"
    assigned = {q: bb.FORMAT for q in full[:-1]}
    assigned[full[-1]] = "BF16"

    arm = bb.run_arm(state, calib, full, render_assignment=assigned)
    rows, byts = bb.compare(arm, reference)
    assert len(arm["weights"]) == len(full) - 1
    assert not rows and not byts, (
        f"one BF16 unit moved {len(byts)} of {len(arm['weights'])} units' "
        "bytes; the hook set is being narrowed again"
    )


def test_shortening_the_enumeration_still_moves_the_bytes(bb, frozen):
    """The one narrowing that is still byte-moving, pinned deliberately.

    ``qnames`` IS the hook set, so a caller who shortens it renders different
    bytes -- by construction, not by defect. Callers narrow with
    ``render_assignment``/``render_qnames`` instead. This test asserts the
    hazard is still real so the docstring warning stays load-bearing; if a
    future change makes a shortened enumeration safe too, it turns red and
    should be deleted with a note rather than left passing silently.
    """
    state, calib, reference = frozen
    arm = bb.run_arm(state, calib, bb.layer_qnames((1, 3)))
    rows, byts = bb.compare(arm, reference)
    n = len(arm["weights"])
    assert n == 4
    assert len(rows) == n and len(byts) == n, (
        f"shortening qnames moved {len(byts)}/{n} units' bytes, not all of "
        "them. The hook set's coupling changed shape; re-measure with "
        f"{_EXPERIMENT.name} before editing this test."
    )


def test_the_cache_records_which_enumeration_it_hooked(bb, frozen):
    """Provenance a reader can use: render_scope alone cannot say.

    A stripe and a whole build both stamp ``render_scope="format-menu"``, and
    ``union_production_cache._render_identity`` binds that string -- so two
    caches rendered against different row sets used to produce the same render
    identity. The hooked-enumeration digest is what distinguishes them.
    """
    state, calib, reference = frozen
    assert reference["hook_scope"] is not None, "no activation_hook_scope stamp"
    whole = reference["hook_scope"]
    assert whole["hooked_qnames"] == len(bb.all_qnames())
    assert whole["render_narrowed"] is False

    stripe = bb.layer_qnames((1, 3))
    arm = bb.run_arm(state, calib, bb.all_qnames(), render_qnames=stripe)
    shard = arm["hook_scope"]
    assert shard["render_narrowed"] is True
    assert shard["rendered_qnames"] == len(stripe)
    # Equal hook digests + equal calibration is the readable claim "these two
    # caches were rendered against the same rows".
    assert shard["hooked_qnames_sha256"] == whole["hooked_qnames_sha256"]

    narrowed = bb.run_arm(state, calib, stripe)
    assert (
        narrowed["hook_scope"]["hooked_qnames_sha256"]
        != whole["hooked_qnames_sha256"]
    ), "a shortened enumeration must be visible in the stamp"


def test_the_hook_set_is_the_full_enumeration_at_both_fixed_sites():
    """Pin both fix sites, so a refactor cannot re-narrow them quietly."""
    import inspect

    import prismaquant.build_production_cache as bpc
    import prismaquant.production_weight_cache as pwc

    fill = inspect.getsource(pwc.fill_production_weight_cache)
    assert "qnames=eligible_qnames," in fill, (
        "the activation collector no longer hooks the full eligible "
        "enumeration; #130/#135 have regressed. Re-run "
        "experiments/stripe_row_identity_byte_baseline.py"
    )
    assert "qnames=qname_set," not in fill, (
        "the collector hooks the render-narrowed qname_set again (#135)"
    )

    main = inspect.getsource(bpc.main)
    assert "qnames = [q for q in qnames if q in allowed]" not in main, (
        "--include-qnames-file shortens the enumeration handed to "
        "fill_production_weight_cache again, so a stripe renders different "
        "rows than an unsharded build (#130)"
    )
    assert "render_qnames=render_only," in main, (
        "--include-qnames-file no longer narrows via render_qnames; re-check "
        "how the stripe's render is scoped"
    )
