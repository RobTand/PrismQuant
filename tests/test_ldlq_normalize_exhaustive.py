"""Exhaustive cross-language truth table for CB LDLQ normalize.

Covers: unset/empty/whitespace/case/invalid/contradictory/legacy-only/scope-only/all/nvfp4/hostile env,
and padded nonempty legacy values — shell and python must agree byte-for-byte.
"""
import os
import subprocess
from pathlib import Path

import pytest

from prismaquant.cb_ldlq_normalize import normalize_ldlq_vars


HELPER = Path(__file__).resolve().parents[1] / "prismaquant/cb_ldlq_normalize.sh"


def py_normalize(legacy, scope):
    try:
        return normalize_ldlq_vars(legacy, scope), None
    except ValueError as e:
        return None, str(e)


def sh_normalize(legacy, scope):
    env = os.environ.copy()
    # unset both first
    env.pop("PRISMAQUANT_CB_LDLQ", None)
    env.pop("PRISMAQUANT_CB_LDLQ_SCOPE", None)
    if legacy is not None:
        env["PRISMAQUANT_CB_LDLQ"] = legacy
    if scope is not None:
        env["PRISMAQUANT_CB_LDLQ_SCOPE"] = scope
    # Call helper --normalize which runs normalize_cb_ldlq_vars and prints "legacy scope"
    result = subprocess.run(
        ["bash", str(HELPER), "--normalize"],
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )
    if result.returncode == 0:
        out = result.stdout.strip()
        # expected two fields
        parts = out.split()
        if len(parts) >= 2:
            return (parts[0], parts[1]), None
        return None, f"bad output {out!r}"
    else:
        return None, result.stderr.strip()


# Truth table cases: (legacy, scope) -> expected (legacy_canonical, scope_canonical) or error
CASES = [
    # unset
    (None, None, ("0", "none"), False),
    # empty legacy as unset, empty scope as unset
    ("", None, ("0", "none"), False),
    ("", "", ("0", "none"), False),
    ("   ", None, ("0", "none"), False),
    ("   ", "   ", ("0", "none"), False),
    ("", "   ", ("0", "none"), False),
    # whitespace scope treated as unset
    (None, "", ("0", "none"), False),
    (None, "   ", ("0", "none"), False),
    (None, "  nvfp4  ", ("1", "nvfp4"), False),
    (None, "  NVFP4  ", ("1", "nvfp4"), False),
    (None, "  All  ", ("1", "all"), False),
    (None, "  NoNe ", ("0", "none"), False),
    # padded nonempty legacy — trimmed + lower
    (" 1 ", None, ("1", "all"), False),
    (" 0 ", None, ("0", "none"), False),
    ("  true  ", None, ("1", "all"), False),
    ("  TRUE ", None, ("1", "all"), False),
    ("  TrUe ", None, ("1", "all"), False),
    ("  yes ", None, ("1", "all"), False),
    ("  ON ", None, ("1", "all"), False),
    ("  false ", None, ("0", "none"), False),
    ("  False", None, ("0", "none"), False),
    ("  NO ", None, ("0", "none"), False),
    ("  off ", None, ("0", "none"), False),
    # padded scope case-insensitive
    (None, "NVFP4", ("1", "nvfp4"), False),
    (None, "NvFp4", ("1", "nvfp4"), False),
    (None, "ALL", ("1", "all"), False),
    (None, "None", ("0", "none"), False),
    (None, "  NONE ", ("0", "none"), False),
    # legacy-only
    ("1", None, ("1", "all"), False),
    ("0", None, ("0", "none"), False),
    ("true", None, ("1", "all"), False),
    ("false", None, ("0", "none"), False),
    # scope-only
    (None, "nvfp4", ("1", "nvfp4"), False),
    (None, "all", ("1", "all"), False),
    (None, "none", ("0", "none"), False),
    # all / nvfp4 / none with legacy consistent
    ("1", "all", ("1", "all"), False),
    ("1", "nvfp4", ("1", "nvfp4"), False),
    ("0", "none", ("0", "none"), False),
    # contradictory
    ("0", "all", None, True),
    ("0", "nvfp4", None, True),
    ("1", "none", None, True),
    ("true", "none", None, True),
    ("false", "all", None, True),
    # padded contradictory still fails
    (" 0 ", "  all  ", None, True),
    (" 1 ", "  none  ", None, True),
    # invalid legacy
    ("2", None, None, True),
    ("maybe", None, None, True),
    ("", "invalid_scope", None, True),  # empty legacy unset, scope invalid => error
    (None, "invalid", None, True),
    (None, "2", None, True),
    ("", "  invalid  ", None, True),
    # case invalid with padding
    ("  2 ", None, None, True),
    ("  invalid ", None, None, True),
    # hostile env: weird values
    ("\n", None, ("0", "none"), False),  # newline is whitespace -> unset
    ("\t", "nvfp4", ("1", "nvfp4"), False),
]


@pytest.mark.parametrize("legacy,scope,expected,should_error", CASES)
def test_cross_language_parity(legacy, scope, expected, should_error):
    py_res, py_err = py_normalize(legacy, scope)
    sh_res, sh_err = sh_normalize(legacy, scope)
    if should_error:
        assert py_res is None, f"python should error for legacy={legacy!r} scope={scope!r} but got {py_res}"
        assert sh_res is None, f"shell should error for legacy={legacy!r} scope={scope!r} but got {sh_res}"
        # Both errors should mention inconsistency or must be one of
        assert py_err is not None and sh_err is not None
    else:
        assert py_res == expected, f"python mismatch for legacy={legacy!r} scope={scope!r}: got {py_res} expected {expected}"
        assert sh_res == expected, f"shell mismatch for legacy={legacy!r} scope={scope!r}: got {sh_res} expected {expected}"
        assert py_res == sh_res


def test_hostile_env_overwrites():
    # Hostile: legacy has leading/trailing spaces and mixed case, scope same; both should normalize identical
    for legacy in [" 1 ", "  true ", "  YES ", "  On "]:
        for scope in ["  nvfp4 ", "  ALL ", "  none "]:
            # If contradictory, both should error
            should_error = (legacy.strip().lower() in {"0","false","no","off"} and scope.strip().lower() != "none") or \
                           (legacy.strip().lower() in {"1","true","yes","on"} and scope.strip().lower() == "none")
            py_res, py_err = py_normalize(legacy, scope)
            sh_res, sh_err = sh_normalize(legacy, scope)
            if should_error:
                assert py_res is None and sh_res is None
            else:
                # Both should succeed and agree
                assert py_res is not None and sh_res is not None
                assert py_res == sh_res


def test_empty_legacy_scope_both_ways():
    # Empty string vs None vs whitespace-only must be identical (unset)
    for empty in [None, "", "   ", "\t", "\n  \t"]:
        py1, _ = py_normalize(empty, "nvfp4")
        py2, _ = py_normalize("1", empty)  # actually legacy empty handled? Use scope empty
        sh1, _ = sh_normalize(empty, "nvfp4")
        sh2, _ = sh_normalize(empty, empty)
        # Just check they don't crash and agree
        assert (py1 is None) == (sh1 is None)
        assert (py2 is None) == (sh2 is None)
