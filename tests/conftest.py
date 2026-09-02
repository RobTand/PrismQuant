"""Shared pytest fixtures.

Deliberately minimal: this repo had no conftest before 2026-08-30. It carried
one NON-autouse fixture, ``synthetic_cb_target``, which let a codebook export
test declare that its bodies were CPU fixtures rather than served artifacts,
through the route-status gate's own ``PQ_CB_NON_NATIVE_TARGET`` declaration.
That gate, that declaration and the export tests that used it all went into
``archive/gridbook_lane_2026-09-02/`` when the Gridbook codebook serving lane
was retired on 2026-09-02, so the fixture has no subject left. Nothing here
was ever autouse, so collection semantics are unchanged either way.
"""
from __future__ import annotations
