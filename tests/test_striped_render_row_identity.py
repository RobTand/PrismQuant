"""A narrowed production-cache render must keep the unstriped run's rows.

Rows feed the GPTQ Hessian; the Hessian feeds the rendered bytes. So a build
that narrows which Linears the collector hooks renders different bytes than a
full build of the same recipe -- two artifacts wearing one name, which is
exactly what principle 8 (one rendering behind the surrogate, the KL and the
exported bytes) exists to prevent.

``56c765d`` closed the ``--resume`` half by drawing reservoir priorities for
every *hooked* Linear. Two narrowings reach the **hook** set itself and are
therefore out of its reach:

  * ``build_production_cache.py:843`` -- ``--include-qnames-file``, i.e. every
    ``production_cache_stripes`` stripe;
  * ``production_weight_cache.py:4888`` + ``:5164`` -- ``qname_set`` is the
    assignment-narrowed render set AND the collector's hook set, so every
    ``--render-scope assignment`` build (``run-pipeline.sh:590``, the shipping
    default) hooks fewer Linears than a ``format-menu`` build of the same model.

Issue #130. The gap is a *ratchet*, not a bare xfail, per this repo's
convention: the divergence tests first assert the gap is still real and only
then xfail, so closing it turns them red with an instruction rather than
passing silently. ``test_widening_the_hook_set_restores_the_unstriped_bytes``
is a plain passing test and pins the shape of the fix.

The full digest matrix, the N_CALIB discriminator (bin shape vs shared stream)
and the machine-readable receipt live in
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


@pytest.mark.parametrize(
    "label,indices",
    [
        ("non-prefix stripe (the plan_stripes LPT bin shape)", (1, 3)),
        ("prefix stripe (a contiguous partition)", (0, 1)),
    ],
)
def test_striped_render_keeps_the_unstriped_rows(bb, frozen, label, indices):
    """RATCHET (#130): narrowing the qnames argument must not move the bytes."""
    state, calib, reference = frozen
    arm = bb.run_arm(state, calib, bb.layer_qnames(indices))
    rows, byts = bb.compare(arm, reference)
    n = len(arm["weights"])
    assert n == 4, "arm did not render the expected unit count"
    if not byts:
        return  # gap closed on this arm; the ratchet has nothing to xfail
    assert len(rows) == n and len(byts) == n, (
        f"#130 changed shape on {label}: {len(rows)}/{n} units keep different "
        f"rows and {len(byts)}/{n} render different bytes. Re-measure with "
        f"{_EXPERIMENT.name} before editing this ratchet."
    )
    pytest.xfail(
        f"#130 open: {label} -- {len(byts)}/{n} units render different bytes "
        "than the unstriped run (build_production_cache.py:843)"
    )


def test_assignment_render_scope_keeps_the_format_menu_rows(bb, frozen):
    """RATCHET (#130): the shipping default's scope switch must not move bytes.

    Same ``qnames`` argument in both arms. The only difference is whether the
    narrowing arrives as ``render_assignment``, which ``:4888`` folds into the
    collector's hook set at ``:5164``.
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
    n = len(arm["weights"])
    assert n == len(stripe)
    if not byts:
        return
    assert len(rows) == n and len(byts) == n, (
        f"#130 changed shape on the assignment scope: {len(rows)}/{n} rows, "
        f"{len(byts)}/{n} bytes. Re-measure before editing this ratchet."
    )
    pytest.xfail(
        f"#130 open: --render-scope assignment renders {len(byts)}/{n} units "
        "differently from --render-scope format-menu "
        "(production_weight_cache.py:4888 -> :5164)"
    )


def test_widening_the_hook_set_restores_the_unstriped_bytes(bb, frozen):
    """The fix's shape, priced by rendering rather than argued.

    Hook the full eligible enumeration and let only the RENDER narrow, and both
    narrowed arms reproduce the unstriped bytes exactly. This is the
    byte-preserving option; a per-qname derived seed would also decouple the
    streams but moves every fresh unstriped build's bytes as well.
    """
    state, calib, reference = frozen
    full = bb.all_qnames()
    stripe = set(bb.layer_qnames((1, 3)))
    assigned = {q: (bb.FORMAT if q in stripe else "BF16") for q in full}

    for label, qnames in (
        ("assignment scope", full),
        ("stripe as a render scope", sorted(stripe)),
    ):
        arm = bb.run_arm(
            state, calib, qnames,
            render_assignment=assigned,
            hook_qnames=set(full),
        )
        rows, byts = bb.compare(arm, reference)
        assert len(arm["weights"]) == len(stripe)
        assert not rows, f"{label}: {len(rows)} units kept different rows"
        assert not byts, f"{label}: {len(byts)} units rendered different bytes"


def test_the_hook_set_is_the_render_narrowed_set_today(bb):
    """Pin the line the ratchets blame, so a refactor cannot silently move it."""
    import inspect

    import prismaquant.production_weight_cache as pwc

    source = inspect.getsource(pwc.fill_production_weight_cache)
    assert "qname_set = set(render_formats_by_qname)" in source, (
        "the assignment-narrowing at production_weight_cache.py:4888 moved; "
        "re-locate it before trusting these ratchets"
    )
    assert "qnames=qname_set," in source, (
        "the collector no longer hooks qname_set; #130 may be fixed or moved. "
        "Re-run experiments/stripe_row_identity_byte_baseline.py"
    )
