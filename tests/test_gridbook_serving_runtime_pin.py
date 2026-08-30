from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from prismaquant.gridbook_serving_runtime_pin import (
    GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING,
    GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA,
    GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
    GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING,
    GridbookServingRuntimePinError,
    load_gridbook_serving_runtime_pin,
    parse_gridbook_serving_runtime_pin,
    require_exact_gridbook_serving_runtime_release,
)


def _resolved_payload() -> dict:
    return {
        "schema": "prismaquant.gridbook_serving_runtime_pin.v1",
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "a" * 40,
        "version": GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
        "version_is_release": True,
        "wheel_sha256": "b" * 64,
        "runtime_contract_schema": GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA,
        "required_abi_features": {
            "routed_moe_per_role_codebook_lut": 1,
            "source_fp8_block128_w8a16": 1,
            "dspark_construction_physical_bridge": 1,
        },
    }


def test_packaged_serving_pin_is_resolved_and_loads_in_shell():
    """The packaged pin resolves, and the shell helper accepts it.

    Until 2026-08-14 this asserted the opposite -- that the packaged pin was
    the PENDING sentinel and that the shell helper exited 2 on it. That was
    the correct assertion while 0.8.6 was untagged. Now that the release
    exists, the same two code paths are asserted from the other side: the pin
    must resolve, and the helper that every serve script sources must export
    the resolved identity rather than refuse. The refusal path is not dropped
    -- it is exercised on a synthetic pending pin in the test below, which is
    where it belongs, since a fixture cannot go stale the way the packaged
    file just did.
    """
    pin = load_gridbook_serving_runtime_pin()
    assert pin.commit != GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING
    assert pin.wheel_sha256 != GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING
    assert pin.commit_is_resolved and pin.wheel_is_resolved
    assert pin.version == GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION
    assert pin.version_is_release is True
    require_exact_gridbook_serving_runtime_release(pin)

    helper = (
        Path(__file__).resolve().parents[1]
        / "prismaquant"
        / "gridbook_runtime"
        / "gridbook_serving_runtime.sh"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; gridbook_serving_runtime_load_pin && printf "%s\\t%s\\t%s\\n"'
            ' "$GRIDBOOK_RUNTIME_COMMIT" "$GRIDBOOK_RUNTIME_VERSION"'
            ' "$GRIDBOOK_RUNTIME_WHEEL_SHA256"',
            "bash",
            str(helper),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    # The shell and Python readers must agree; they are two independent parsers
    # of the same file and a serve script only ever sees the shell one.
    assert result.stdout.strip().split("\t") == [
        pin.commit, pin.version, pin.wheel_sha256,
    ]


def test_shell_helper_still_refuses_a_pending_serving_pin(tmp_path):
    """A pending pin must still fail the shell loader closed, exit 2."""
    helper = (
        Path(__file__).resolve().parents[1]
        / "prismaquant"
        / "gridbook_runtime"
        / "gridbook_serving_runtime.sh"
    )
    asset_dir = tmp_path / "gridbook_runtime"
    asset_dir.mkdir()
    shutil.copy(helper, asset_dir / "gridbook_serving_runtime.sh")
    payload = json.loads(
        (helper.parent / "gridbook_serving_runtime_pin.json")
        .read_text(encoding="utf-8"))
    payload["commit"] = GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING
    payload["wheel_sha256"] = GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING
    payload["version_is_release"] = False
    (asset_dir / "gridbook_serving_runtime_pin.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; gridbook_serving_runtime_load_pin',
            "bash",
            str(asset_dir / "gridbook_serving_runtime.sh"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 2
    assert "invalid or pending serving pin" in result.stderr


def test_resolved_serving_pin_requires_closed_v4_feature_set():
    pin = parse_gridbook_serving_runtime_pin(_resolved_payload())
    assert pin.commit_is_resolved
    assert pin.wheel_is_resolved
    bad = deepcopy(_resolved_payload())
    bad["required_abi_features"].pop("dspark_construction_physical_bridge")
    with pytest.raises(GridbookServingRuntimePinError, match="closure differs"):
        parse_gridbook_serving_runtime_pin(bad)


def test_serving_helper_accepts_only_the_exact_resolved_wheel(tmp_path):
    wheel = tmp_path / f"gridbook-{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"gridbook-{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: gridbook\nVersion: "
            f"{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}\n",
        )
        archive.writestr("gridbook/__init__.py", "")
    payload = _resolved_payload()
    payload["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    source = (
        Path(__file__).resolve().parents[1]
        / "prismaquant"
        / "gridbook_runtime"
        / "gridbook_serving_runtime.sh"
    )
    helper = runtime_dir / source.name
    shutil.copyfile(source, helper)
    (runtime_dir / "gridbook_serving_runtime_pin.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; gridbook_serving_runtime_load_pin; '
            'gridbook_serving_runtime_verify_wheel "$2"',
            "bash",
            str(helper),
            str(wheel),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(wheel.resolve())

    payload["wheel_sha256"] = "b" * 64
    (runtime_dir / "gridbook_serving_runtime_pin.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    rejected = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; gridbook_serving_runtime_load_pin; '
            'gridbook_serving_runtime_verify_wheel "$2"',
            "bash",
            str(helper),
            str(wheel),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode != 0
    assert "differs from pin" in rejected.stderr


def _serving_helper_copy(tmp_path, payload):
    """Hermetic copy of the serving helper beside a chosen pin payload."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(exist_ok=True)
    source = (
        Path(__file__).resolve().parents[1]
        / "prismaquant"
        / "gridbook_runtime"
        / "gridbook_serving_runtime.sh"
    )
    helper = runtime_dir / source.name
    shutil.copyfile(source, helper)
    (runtime_dir / "gridbook_serving_runtime_pin.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return helper


def _pip_download_shim(tmp_path, served_wheel):
    """A ``python3`` that answers ``-m pip download`` with a chosen wheel.

    The defect under test only fires on the DOWNLOAD path: with a *supplied*
    wheel a rejected artifact leaves the staging directory empty, so the
    materializer refuses on the "did not yield exactly one wheel" check and
    never reaches the publish.  pip must therefore actually deliver a wheel
    that the pin rejects -- which is precisely what PyPI did on 2026-08-14.
    Everything other than ``pip download`` delegates to the real interpreter,
    because the helper also runs Python to load the pin and hash the wheel.
    """
    bin_dir = tmp_path / "shim-bin"
    bin_dir.mkdir(exist_ok=True)
    shim = bin_dir / "python3"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"pip download"* ]]; then\n'
        '  dest=""; prev=""\n'
        '  for a in "$@"; do\n'
        '    if [[ "$prev" == "--dest" ]]; then dest="$a"; fi\n'
        '    prev="$a"\n'
        "  done\n"
        '  cp -- "$PQ_TEST_SERVED_WHEEL" "$dest/"\n'
        "  exit 0\n"
        "fi\n"
        'exec "$PQ_TEST_REAL_PYTHON3" "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return bin_dir, {
        "PQ_TEST_SERVED_WHEEL": str(served_wheel),
        "PQ_TEST_REAL_PYTHON3": shutil.which("python3") or sys.executable,
    }


def _materialize_in_caller_shape(
    helper, cache_dir, supplied=None, shim=None
):
    """Run the materializer through its REAL caller's command shape.

    ``gridbook_serving_runtime_prepare`` reaches the materializer as
    ``wheel="$(_gridbook_serving_materialize_wheel)" || return``.  Bash
    disables errexit for a command substitution inside a ``||`` list, so the
    ``set -euo pipefail`` in the materializer's own subshell is inert on this
    path.  Invoking the function any other way would silently restore errexit
    and make this test pass against the very defect it exists to catch.
    """
    environment = {
        **os.environ,
        "GRIDBOOK_SERVING_RUNTIME_CACHE_DIR": str(cache_dir),
    }
    if supplied is not None:
        environment["GRIDBOOK_SERVING_RUNTIME_WHEEL"] = str(supplied)
    else:
        environment.pop("GRIDBOOK_SERVING_RUNTIME_WHEEL", None)
    if shim is not None:
        bin_dir, shim_environment = shim
        environment.update(shim_environment)
        environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    return subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; gridbook_serving_runtime_load_pin; '
            "caller() { local w; "
            'w="$(_gridbook_serving_materialize_wheel)" || return; '
            'printf %s\\\\n "$w"; }; caller',
            "bash",
            str(helper),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )


def test_a_rejected_wheel_is_never_published_into_the_digest_cache(tmp_path):
    """A wheel the pin rejects must not be cached under the pinned digest.

    Regression for a defect that BRICKED the DSpark serving lane on
    2026-08-14.  The wheel cache is keyed by the pinned digest, and the
    materializer's first branch trusts that directory name: it takes the one
    wheel inside and verifies it, without ever consulting a supplied wheel or
    a download.  So caching an unverified wheel is not a wasted byte -- it is
    permanent.  Every later invocation finds the wrong wheel, refuses, and no
    correct wheel can be introduced through this path again.

    That is exactly what happened: gridbook 0.8.6 was published to PyPI, a
    prepare without a supplied wheel downloaded it, and although it does NOT
    satisfy the pin (the pin names the wheel read out of the served image;
    the PyPI archive is content-identical but a different artifact) it was
    still moved into the digest directory, refusing the lane until the
    directory was removed by hand.
    """
    good = tmp_path / f"gridbook-{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr(
            f"gridbook-{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: gridbook\nVersion: "
            f"{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}\n",
        )
        archive.writestr("gridbook/__init__.py", "")
    good_digest = hashlib.sha256(good.read_bytes()).hexdigest()

    # A DIFFERENT archive carrying the same name/version -- the shape of the
    # PyPI-vs-served-image mismatch that caused the incident.
    other = tmp_path / "other" / f"gridbook-{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}-py3-none-any.whl"
    other.parent.mkdir()
    with zipfile.ZipFile(other, "w") as archive:
        archive.writestr(
            f"gridbook-{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: gridbook\nVersion: "
            f"{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}\n",
        )
        archive.writestr("gridbook/__init__.py", "# rebuilt\n")
    assert hashlib.sha256(other.read_bytes()).hexdigest() != good_digest

    payload = _resolved_payload()
    payload["wheel_sha256"] = good_digest
    helper = _serving_helper_copy(tmp_path, payload)
    cache_dir = tmp_path / "cache"
    destination = cache_dir / good_digest

    # Drive the DOWNLOAD path, the one the incident took: pip delivers a
    # wheel the pin rejects.  A supplied mismatched wheel would exit earlier
    # on the empty-staging-directory check and prove nothing.
    rejected = _materialize_in_caller_shape(
        helper, cache_dir, shim=_pip_download_shim(tmp_path, other)
    )
    assert rejected.returncode != 0, rejected.stdout
    assert not destination.exists(), (
        "a wheel that failed digest verification was published into the "
        "digest-named cache, which permanently bricks the serving lane"
    )

    # A rejected SUPPLIED wheel must also leave no cache behind.
    supplied_rejected = _materialize_in_caller_shape(helper, cache_dir, other)
    assert supplied_rejected.returncode != 0
    assert not destination.exists()

    # Positive control: the pinned artifact still populates the cache, so the
    # guard rejects the wrong wheel rather than disabling caching outright.
    accepted = _materialize_in_caller_shape(helper, cache_dir, good)
    assert accepted.returncode == 0, accepted.stderr
    assert destination.is_dir()
    assert [item.name for item in destination.iterdir()] == [good.name]


def test_a_poisoned_cache_entry_names_the_directory_to_remove(tmp_path):
    """The refusal must tell the operator WHICH cache directory is at fault.

    The fast path short-circuits before any supplied wheel is read, so an
    operator told only "SHA-256 X differs from pin Y" will keep supplying a
    correct wheel that is never consulted.  Naming the directory is what
    makes the state recoverable.
    """
    good = tmp_path / f"gridbook-{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr(
            f"gridbook-{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: gridbook\nVersion: "
            f"{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}\n",
        )
        archive.writestr("gridbook/__init__.py", "")
    payload = _resolved_payload()
    payload["wheel_sha256"] = hashlib.sha256(good.read_bytes()).hexdigest()
    helper = _serving_helper_copy(tmp_path, payload)

    cache_dir = tmp_path / "cache"
    destination = cache_dir / payload["wheel_sha256"]
    destination.mkdir(parents=True)
    with zipfile.ZipFile(
        destination / f"gridbook-{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}-py3-none-any.whl", "w"
    ) as archive:
        archive.writestr(
            f"gridbook-{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: gridbook\nVersion: "
            f"{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}\n",
        )
        archive.writestr("gridbook/__init__.py", "# poisoned\n")

    result = _materialize_in_caller_shape(helper, cache_dir, good)
    assert result.returncode != 0
    assert str(destination) in result.stderr, result.stderr


@pytest.mark.parametrize("member", ("commit", "wheel_sha256"))
def test_pending_serving_identity_cannot_claim_release(member):
    payload = _resolved_payload()
    payload[member] = (
        GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING
        if member == "commit"
        else GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING
    )
    with pytest.raises(GridbookServingRuntimePinError, match="cannot be marked"):
        parse_gridbook_serving_runtime_pin(payload)
