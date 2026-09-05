"""Shared pytest fixtures.

Formerly minimal: this repo had no conftest before 2026-08-30. It carried one
NON-autouse fixture, ``synthetic_cb_target``, which let a codebook export test
declare that its bodies were CPU fixtures rather than served artifacts, through
the route-status gate's own ``PQ_CB_NON_NATIVE_TARGET`` declaration. That gate,
that declaration and the export tests that used it all went into
``archive/gridbook_lane_2026-09-02/`` when the Gridbook codebook serving lane
was retired on 2026-09-02, so the fixture has no subject left.

What is here now is one autouse fixture, which exists because profile
detection is process-global (issue #197; see its docstring), and the
legacy-lane-grammar fixtures (``legacy_v4_contract``, ``legacy_v5_contract``,
``down_convert_lane_table``), which exist so a test about an OLDER Tessera
lane grammar owns its fixture instead of asserting that the installed
contract is a version it no longer is.
"""
from __future__ import annotations

import copy
import json

import pytest


def _installed_contract() -> dict:
    """The Tessera contract the environment installs, parsed."""
    from importlib.resources import as_file

    from prismaquant import tessera_runtime_contract as contract

    with as_file(contract.contract_path()) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def down_convert_lane_table(payload: dict, schema: str) -> dict:
    """The installed contract, expressed in an OLDER lane grammar.

    Legacy-grammar tests need a legacy table. They used to get one by reading
    the installed contract, which was v4 at the time -- so on the day Tessera
    shipped ``tessera.lane-eligibility.v6`` (its PR #176) every such test began
    asserting that the installed contract was a version it no longer is. That
    is a fixture problem, not a reader problem: a test about a legacy grammar
    must OWN its legacy fixture, exactly as ``test_tessera_contract_v5`` already
    did by rewriting ``runtime`` onto every cell.

    Down-converting rather than hand-writing keeps the fixture honest about
    everything the grammar did not change -- families, rungs, route statuses,
    launches -- so the test still exercises real cells. It is a FIXTURE and
    never an attestation: nothing derived from it is recorded anywhere.

    ``schema`` is ``tessera.lane-eligibility.v8`` (drop the v9
    ``smoke.record``), ``...v7`` (drop the v8 ``evidence.artifact`` as well),
    ``...v6`` (drop v7's ``smoke.attribution`` and ``smoke.control`` too),
    ``...v5`` (drop the whole ``evidence`` block and the ``runtime`` version
    fields) or ``...v4`` (drop the per-cell ``runtime`` scope as well).

    Dropping v9's record is not only a key removal: v9 DERIVES
    ``smoke.attribution`` from the record, and v7/v8 derive it from the
    control.  A fixture that removed the record and kept the derived
    attribution would be a table no validator would accept, so the
    attribution is re-derived under the older rule -- through the reader's own
    ``derive_smoke_attribution``, which is that rule's home here.
    """
    from prismaquant.lane_eligibility import (
        EVIDENCE_ATTRIBUTION_UNATTRIBUTED, EVIDENCE_OUTCOME_IDENTICAL,
        EVIDENCE_ATTRIBUTION_SHARED, EVIDENCE_ATTRIBUTION_NOT_SHARED)

    payload = copy.deepcopy(payload)
    lane = payload["lane_eligibility"]
    lane["schema"] = schema
    version = int(schema.rsplit(".v", 1)[1])
    for cell in lane["cells"]:
        if version <= 5:
            cell.pop("evidence", None)
        else:
            evidence = cell["evidence"]
            if version <= 8:
                smoke = evidence["smoke"]
                if smoke.pop("record", None) is not None:
                    control = smoke.get("control")
                    smoke["attribution"] = (
                        EVIDENCE_ATTRIBUTION_UNATTRIBUTED if control is None
                        else EVIDENCE_ATTRIBUTION_SHARED
                        if control["outcome"] == EVIDENCE_OUTCOME_IDENTICAL
                        else EVIDENCE_ATTRIBUTION_NOT_SHARED)
            if version <= 7:
                evidence.pop("artifact", None)
            if version <= 6:
                evidence["smoke"].pop("attribution", None)
                evidence["smoke"].pop("control", None)
        if version <= 4:
            cell.pop("runtime", None)
        elif version <= 5:
            runtime = cell.get("runtime", {})
            cell["runtime"] = {"image": runtime["image"],
                               "execution_modes": runtime["execution_modes"]}
    return payload


@pytest.fixture
def legacy_v4_contract() -> dict:
    """The installed contract expressed as a v4 lane table."""
    return down_convert_lane_table(_installed_contract(),
                                   "tessera.lane-eligibility.v4")


@pytest.fixture
def legacy_v5_contract() -> dict:
    """The installed contract expressed as a v5 lane table."""
    return down_convert_lane_table(_installed_contract(),
                                   "tessera.lane-eligibility.v5")


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
