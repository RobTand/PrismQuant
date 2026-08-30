"""Closed serving-environment contract for the pinned Gridbook release."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from prismaquant import gridbook_environment as envmod


EXECUTION = (
    "GRIDBOOK_MXFP8_DENSE",
    "PRISMAQUANT_CB_GEMV",
    "PRISMAQUANT_CB_FP8_GEMV_V2",
    "PRISMAQUANT_CB_FUSED_FP4",
    "PRISMAQUANT_CB_FUSED_FP4_MOE",
    "PRISMAQUANT_CB_BF16_SM120",
    "PRISMAQUANT_CB_FP4_FUSED_MIDM",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R",
    "PRISMAQUANT_CB_FUSED_MIDM",
    "PRISMAQUANT_CB_GROUPED_TRIM",
    "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK",
    "PRISMAQUANT_CB_PREFILL_CHUNK_BYTES",
    "PRISMAQUANT_CB_DECODE_CONTRACT",
    "PRISMAQUANT_CB_FP8_SCHED",
    "PRISMAQUANT_CB_FP4V2_SCHED",
    "PRISMAQUANT_CB_FP4V2_DENSE_R2",
    "PRISMAQUANT_CB_W2_SCHED",
    "PRISMAQUANT_CB_W2_ROWS",
    "PRISMAQUANT_CB_W2_WARPS",
    "VLLM_USE_DEEP_GEMM",
)
CORRECTNESS_BYPASS = ("PRISMAQUANT_SKIP_CB_CAST_CHECK",)
RESIDENCY_BUILD = (
    "PRISMAQUANT_PRELOAD_FUSED",
    "PRISMAQUANT_CB_EXT_DIR",
    "PRISMAQUANT_CUTLASS_INCLUDE",
    "CUDACXX",
    "CXX",
)
RETIRED = (
    "PRISMAQUANT_CB_DECODE",
    "PRISMAQUANT_CB_EXPAND",
    "PRISMAQUANT_CB_PREFILL",
)
DIAGNOSTIC = ("PRISMAQUANT_DEBUG_PREFIXES",)

CANONICAL = {
    "CUDACXX": None,
    "CXX": None,
    "GRIDBOOK_MXFP8_DENSE": None,
    "PRISMAQUANT_CB_BF16_SM120": "0",
    "PRISMAQUANT_CB_DECODE": None,
    "PRISMAQUANT_CB_DECODE_CONTRACT": "v1",
    "PRISMAQUANT_CB_EXPAND": None,
    "PRISMAQUANT_CB_EXT_DIR": None,
    "PRISMAQUANT_CB_FP4V2_DENSE_R2": "0",
    "PRISMAQUANT_CB_FP4V2_SCHED": None,
    "PRISMAQUANT_CB_FP4_FUSED_MIDM": "0",
    "PRISMAQUANT_CB_FP8_GEMV_V2": "0",
    "PRISMAQUANT_CB_FP8_SCHED": None,
    "PRISMAQUANT_CB_FUSED_FP4": None,
    "PRISMAQUANT_CB_FUSED_FP4_MOE": None,
    "PRISMAQUANT_CB_FUSED_MIDM": "1",
    "PRISMAQUANT_CB_GEMV": "inherited",
    "PRISMAQUANT_CB_GROUPED_TRIM": "1",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B": "0",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG": "0",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R": "0",
    "PRISMAQUANT_CB_PREFILL": None,
    "PRISMAQUANT_CB_PREFILL_CHUNK_BYTES": "1073741824",
    "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK": None,
    "PRISMAQUANT_CB_W2_ROWS": None,
    "PRISMAQUANT_CB_W2_SCHED": None,
    "PRISMAQUANT_CB_W2_WARPS": None,
    "PRISMAQUANT_CUTLASS_INCLUDE": None,
    "PRISMAQUANT_DEBUG_PREFIXES": None,
    "PRISMAQUANT_PRELOAD_FUSED": "0",
    "PRISMAQUANT_SKIP_CB_CAST_CHECK": "0",
    "VLLM_USE_DEEP_GEMM": "0",
}


def _gridbook_source_root() -> Path:
    configured = os.environ.get("GRIDBOOK_SOURCE_ROOT")
    if not configured:
        pytest.skip(
            "set GRIDBOOK_SOURCE_ROOT to the explicit pinned Gridbook checkout"
        )
    root = Path(configured)
    if not (root / "gridbook" / "lane_select.py").is_file():
        pytest.skip("configured Gridbook source checkout is not present")
    return root


def test_registry_categories_and_allowlist_are_exact_and_disjoint():
    assert envmod.GRIDBOOK_EXECUTION_ENVIRONMENT == EXECUTION
    assert envmod.GRIDBOOK_CORRECTNESS_BYPASS_ENVIRONMENT == CORRECTNESS_BYPASS
    assert envmod.GRIDBOOK_RESIDENCY_BUILD_ENVIRONMENT == RESIDENCY_BUILD
    assert envmod.GRIDBOOK_RETIRED_ENVIRONMENT == RETIRED
    assert envmod.GRIDBOOK_DIAGNOSTIC_ENVIRONMENT == DIAGNOSTIC

    categories = EXECUTION + CORRECTNESS_BYPASS + RESIDENCY_BUILD + RETIRED + DIAGNOSTIC
    assert len(categories) == len(set(categories))
    assert envmod.GRIDBOOK_ENVIRONMENT_ALLOWLIST == tuple(sorted(categories))


def test_canonical_gold_snapshot_is_complete_and_exact():
    assert dict(envmod.CANONICAL_GOLD_ENVIRONMENT) == CANONICAL
    assert dict(envmod.CANONICAL_GOLD_SET_ENVIRONMENT) == {
        name: value for name, value in CANONICAL.items() if value is not None
    }
    assert envmod.CANONICAL_GOLD_CLEARED_ENVIRONMENT == tuple(
        name for name, value in CANONICAL.items() if value is None
    )

    # These three fields cannot safely be represented by a literal "0" in
    # Gridbook 0.8.4.  Pin absence instead of relying on a permissive parser.
    assert CANONICAL["PRISMAQUANT_CB_FUSED_FP4"] is None
    assert CANONICAL["PRISMAQUANT_CB_FUSED_FP4_MOE"] is None
    assert CANONICAL["PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK"] is None


def test_apply_clears_poisoned_state_before_setting_canonical_values():
    environ = {
        **{name: "poison" for name in envmod.GRIDBOOK_ENVIRONMENT_ALLOWLIST},
        "UNRELATED_APPLICATION_SETTING": "preserved",
    }
    observed = envmod.apply_canonical_gold_environment(
        environ, require_pin=False
    )
    assert observed == CANONICAL
    assert environ == {
        "UNRELATED_APPLICATION_SETTING": "preserved",
        **{
            name: value for name, value in CANONICAL.items()
            if value is not None
        },
    }


def test_attestation_receipt_is_full_and_fails_closed_on_set_and_unset_drift():
    environ: dict[str, str] = {}
    envmod.apply_canonical_gold_environment(environ, require_pin=False)
    receipt = envmod.attest_canonical_gold_environment(
        environ, require_pin=False
    )
    assert receipt == {
        "schema": envmod.GRIDBOOK_ENVIRONMENT_SCHEMA,
        "gridbook_version": "0.8.11",
        "gridbook_commit": envmod.PINNED_GRIDBOOK_COMMIT,
        "environment": CANONICAL,
    }

    environ["PRISMAQUANT_CB_FUSED_FP4"] = "0"
    with pytest.raises(
        envmod.GridbookEnvironmentError,
        match=r"PRISMAQUANT_CB_FUSED_FP4: expected <unset>, observed '0'",
    ):
        envmod.attest_canonical_gold_environment(environ, require_pin=False)

    environ.pop("PRISMAQUANT_CB_FUSED_FP4")
    environ.pop("PRISMAQUANT_CB_FUSED_MIDM")
    with pytest.raises(
        envmod.GridbookEnvironmentError,
        match=r"PRISMAQUANT_CB_FUSED_MIDM: expected '1', observed <unset>",
    ):
        envmod.attest_canonical_gold_environment(environ, require_pin=False)


def test_registry_matches_the_packaged_runtime_contract():
    envmod.require_pinned_gridbook_runtime()
    # Deliberately a literal, not the module constant: this test is the second
    # independent witness that the packaged pin is the reviewed release.
    assert envmod.PINNED_GRIDBOOK_COMMIT == (
        "187c7216b9d4882321c1923de0b4c49dc139743c"
    )


def test_registry_rejects_an_alternate_resolved_release(monkeypatch):
    from dataclasses import replace

    alternate = replace(
        envmod.load_gridbook_runtime_pin(),
        commit="a" * 40,
    )
    monkeypatch.setattr(
        envmod,
        "load_gridbook_runtime_pin",
        lambda: alternate,
    )

    with pytest.raises(envmod.GridbookEnvironmentError, match="exact release"):
        envmod.require_pinned_gridbook_runtime()


def test_source_scanner_surfaces_a_new_environment_identifier(tmp_path: Path):
    package = tmp_path / "gridbook"
    package.mkdir()
    (package / "lane_select.py").write_text(
        'import os\nVALUE = os.environ.get("GRIDBOOK_NEW_EXECUTION_FLAG")\n',
        encoding="utf-8",
    )
    report = envmod.scan_gridbook_source_environment(tmp_path)
    assert report.unknown_identifiers == ("GRIDBOOK_NEW_EXECUTION_FLAG",)
    with pytest.raises(
        envmod.GridbookEnvironmentError,
        match="GRIDBOOK_NEW_EXECUTION_FLAG",
    ):
        envmod.require_gridbook_source_compatible(tmp_path)


def test_released_source_environment_scan_is_closed_when_checkout_is_present():
    root = _gridbook_source_root()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == envmod.PINNED_GRIDBOOK_COMMIT

    report = envmod.require_gridbook_source_compatible(root)
    assert report.unknown_identifiers == ()
    assert report.missing_expected_identifiers == ()
    assert report.classified_non_environment == tuple(sorted(
        envmod.GRIDBOOK_SOURCE_NON_ENVIRONMENT_IDENTIFIERS
    ))
    # Two retired switches no longer occur in runtime source.  Every other
    # registry member is visibly accounted for by the source scan.
    assert set(envmod.GRIDBOOK_ENVIRONMENT_ALLOWLIST) - set(
        report.registered_environment
    ) == {"PRISMAQUANT_CB_DECODE", "PRISMAQUANT_CB_EXPAND"}


def test_released_source_domains_explain_every_required_unset_default():
    root = _gridbook_source_root()
    linear = (root / "gridbook" / "linear.py").read_text(encoding="utf-8")
    moe = (root / "gridbook" / "moe.py").read_text(encoding="utf-8")
    lane_select = (root / "gridbook" / "lane_select.py").read_text(
        encoding="utf-8"
    )
    gemv = (root / "gridbook" / "csrc" / "cb_gemv.cu").read_text(
        encoding="utf-8"
    )

    assert '_FP4_FUSED_ALLOWED_MODES = frozenset(("",)) | _FP4_FUSED_MODES' in linear
    assert '_FUSED_FP4_MOE_ALLOWED_MODES = frozenset(("",)) | _FUSED_FP4_MOE_MODES' in moe
    assert '"PRISMAQUANT_CB_FUSED_MIDM", default=True' in linear
    assert '"PRISMAQUANT_CB_GROUPED_TRIM", default=True' in moe
    assert 'default=0, minimum=1' in moe
    assert 'default=(1 << 30), minimum=1' in moe
    assert '_CB_GEMV_VALUES = ("inherited", "auto", "v2")' in (
        root / "gridbook" / "moe_gemv_select.py"
    ).read_text(encoding="utf-8")
    assert '_TRUE = ("1",)' in lane_select
    assert '_FALSE = ("", "0")' in lane_select
    # The two identifiers registered when the producer pin advanced to 0.8.11.
    # The FP8 sibling is a tri-state whose "0" spelling means off; D2R is a
    # plain latched_bool, so "0" is in its domain via _FALSE above.
    assert '_CB_FP8_GEMV_V2_ENV = "PRISMAQUANT_CB_FP8_GEMV_V2"' in (
        root / "gridbook" / "moe_gemv_select.py"
    ).read_text(encoding="utf-8")
    persistent_b = (root / "gridbook" / "moe_persistent_b_lane.py").read_text(
        encoding="utf-8"
    )
    assert '_D2R_FLAG = "PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R"' in persistent_b
    assert "lane_select.latched_bool(\n        _D2R_FLAG," in persistent_b
    assert '!pq_env_is("PRISMAQUANT_CB_FP8_SCHED", "legacy")' in gemv
    assert 'pq_env_is("PRISMAQUANT_CB_FP4V2_SCHED", "db")' in gemv
    assert 'pq_env_is("PRISMAQUANT_CB_W2_SCHED", "legacy")' in gemv
    assert 'pq_env_is("PRISMAQUANT_CB_W2_SCHED", "rowpack")' in gemv
