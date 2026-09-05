"""Shared pytest fixtures.

Formerly minimal: this repo had no conftest before 2026-08-30. It carried one
NON-autouse fixture, ``synthetic_cb_target``, which let a codebook export test
declare that its bodies were CPU fixtures rather than served artifacts, through
the route-status gate's own ``PQ_CB_NON_NATIVE_TARGET`` declaration. That gate,
that declaration and the export tests that used it all went into
``archive/gridbook_lane_2026-09-02/`` when the Gridbook codebook serving lane
was retired on 2026-09-02, so the fixture has no subject left.

What is here now is one autouse fixture, and it exists because profile
detection is process-global (issue #197). See its docstring.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _restore_profile_detection_globals():
    """Snapshot and restore the process-global state ``detect_profile`` reads.

    Profile detection is not a pure function of its arguments. It consults five
    module-level mutables that any test can perturb and that no single test
    owns, and a perturbation that outlives its test silently changes what
    ``detect_profile`` answers for every test after it in that process:

    * ``prismaquant.vendored.OVERRIDE_ERRORS`` -- the dead-vendored-override
      record. ``registry._refuse_dead_vendored_override`` raises on a hit.
      Since issue #201 that raise escapes ``_resolve`` as a
      ``DeadVendoredOverrideError``, so a stray entry now fails the next test
      that detects that architecture, loudly and by name. It used to happen
      *inside* ``_resolve``'s ``except Exception: continue``, which merely
      DEMOTED the profile that matched and let detection fall through to
      ``DefaultProfile`` -- silent, and the reason #197 was so hard to see.
    * ``prismaquant.vendored._QWEN3_REGISTERED`` -- once True, ``register_qwen3``
      returns before the ``OVERRIDE_ERRORS.pop()`` that a successful override
      performs, so a stray ``"qwen3"`` entry can never self-heal.
    * ``registry._REGISTERED``, ``_REGISTRY_GENERATION`` and
      ``_DETECTION_ORDER_CACHE`` -- the registration list, and the derived
      detection order cached against its generation counter.

    Issue #197 was exactly the first two together:
    ``test_vendored_qwen3.py::test_register_qwen3_does_not_cache_a_failed_registration``
    forces ``register_qwen3()`` to fail, ``_fatal()`` records
    ``OVERRIDE_ERRORS["qwen3"]`` on the way past, and nothing puts it back --
    while ``_QWEN3_REGISTERED`` is restored to True, which is what makes the
    entry permanent. Every later ``detect_profile()`` on a qwen3 checkpoint in
    that process then answers ``DefaultProfile``, whose ``structure_spec()`` is
    None, so ``tessera_serving_scope.unit_structure_from_profile`` refuses with
    "explicit Tessera scope needs a declared model profile". That is how
    ``tests/test_tessera_export_scope.py`` came to pass alone and fail in
    company -- on whichever xdist worker happened to draw the two files in that
    order, which is why it was invisible in a serial run (``test_tessera_*``
    sorts before ``test_vendored_*``).

    Restoring is silent and unconditional rather than an assertion. Several
    tests perturb this state deliberately and are right to; the defect is never
    the perturbation, only its escape. Restoring at the one place that owns
    "process-global detection state" fixes the class, where guarding each
    polluter fixes an instance and waits for the next one.

    Cost is a dict copy and a short list copy per test, and the writes only
    happen when something actually changed.
    """
    from prismaquant.model_profiles import registry
    import prismaquant.vendored as vendored

    saved_errors = dict(vendored.OVERRIDE_ERRORS)
    saved_qwen3_registered = vendored._QWEN3_REGISTERED
    saved_registered = list(registry._REGISTERED)
    saved_generation = registry._REGISTRY_GENERATION
    saved_order_cache = registry._DETECTION_ORDER_CACHE
    try:
        yield
    finally:
        if vendored.OVERRIDE_ERRORS != saved_errors:
            vendored.OVERRIDE_ERRORS.clear()
            vendored.OVERRIDE_ERRORS.update(saved_errors)
        vendored._QWEN3_REGISTERED = saved_qwen3_registered
        if registry._REGISTERED != saved_registered:
            # In place: importers bind this list object, not its name.
            registry._REGISTERED[:] = saved_registered
        registry._REGISTRY_GENERATION = saved_generation
        registry._DETECTION_ORDER_CACHE = saved_order_cache
