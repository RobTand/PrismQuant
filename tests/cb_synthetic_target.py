"""One declaration for CB test bodies that are built but never served.

Gridbook 0.9.1's lane-eligibility table (contract v12) names no CB cell on
``sm_121``, and ``serving_profile_specs/nvfp4_cb.json`` targets ``sm_121``
because that is the platform our CB artifacts actually serve on.  So every CB
export now resolves ``unattested`` there and the route-status gate refuses --
correctly.  That refusal is the table honestly reporting a serving gap, not a
gate defect, and it is asserted against the real pin in
``tests/test_cb_route_status_gate.py``.

The synthetic bodies in the CB test modules are a different thing entirely:
five-unit fixtures built on CPU to exercise packing, sharding and decode
geometry, never loaded by a runtime.  They declare what they are, using the
platform's own sanctioned declaration (``PQ_CB_NON_NATIVE_TARGET``), which is
stamped into the export provenance rather than hidden.  This is the artifact
saying "I do not target a native route here", NOT
``PQ_CB_ROUTE_STATUS_OVERRIDE``, which is a decision to ship past a gate and
is not what a test fixture is doing.

Declared per module, visibly, via::

    pytestmark = pytest.mark.usefixtures("synthetic_cb_target")

so it is greppable and no module is silenced by default.
"""
from __future__ import annotations

from contextlib import contextmanager
import os

SYNTHETIC_TARGET_REASON = (
    "synthetic CB test fixture; built on CPU to exercise packing and decode "
    "geometry, never loaded by a serving runtime"
)


@contextmanager
def declare_synthetic_cb_target(reason: str = SYNTHETIC_TARGET_REASON):
    """Declare a non-native target for the duration of the block.

    Set and restored around the block rather than at module import: a bare
    module-level assignment would leak the declaration into every other test
    in the same pytest process, including the ones whose job is to prove the
    gate still refuses.
    """
    # Imported HERE, not at module scope. conftest imports this module at
    # collection start, and a top-level `from prismaquant...` would drag
    # prismaquant/__init__ (the transformers-5.x polyfills, which have known
    # side effects) into every pytest process before any test module runs.
    # This tree has documented order-dependent failures; a pin commit is not
    # the place to move the package import to the front of collection.
    from prismaquant.cb_route_status_gate import NON_NATIVE_TARGET_ENV

    prior = os.environ.get(NON_NATIVE_TARGET_ENV)
    os.environ[NON_NATIVE_TARGET_ENV] = reason
    try:
        yield reason
    finally:
        if prior is None:
            os.environ.pop(NON_NATIVE_TARGET_ENV, None)
        else:
            os.environ[NON_NATIVE_TARGET_ENV] = prior
