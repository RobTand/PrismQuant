from __future__ import annotations

import importlib.util
import json
import select
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/atomic_publication.py"
)
_SPEC = importlib.util.spec_from_file_location("numeric_atomic_publication", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
P = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(P)


def test_claim_excludes_live_competitor_and_same_identity_can_resume(tmp_path):
    destination = tmp_path / "result.json"
    identity = {"schema": "test.v1", "input_sha256": "a" * 64}
    owner_entered = threading.Event()
    owner_release = threading.Event()

    def owner():
        with P.exclusive_publication_claim(destination, identity=identity):
            owner_entered.set()
            assert owner_release.wait(timeout=10)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(owner)
        assert owner_entered.wait(timeout=10)
        with pytest.raises(P.PublicationError, match="competing producer"):
            with P.exclusive_publication_claim(destination, identity=identity):
                pass
        owner_release.set()
        future.result(timeout=10)

    # Kernel-released ownership permits a deterministic crash resume.
    with P.exclusive_publication_claim(destination, identity=identity):
        pass
    with pytest.raises(P.PublicationError, match="identity differs"):
        with P.exclusive_publication_claim(
            destination,
            identity={"schema": "test.v1", "input_sha256": "b" * 64},
        ):
            pass


def test_checkpoint_is_result_last_and_final_publication_never_replaces(tmp_path):
    destination = tmp_path / "result.json"
    partial = tmp_path / "result.json.partial"
    identity = {"schema": "test.v1", "input_sha256": "a" * 64}
    with P.exclusive_publication_claim(destination, identity=identity):
        P.atomic_checkpoint_json(partial, {"partial": True, "rows": 1})
        P.atomic_checkpoint_json(partial, {"partial": False, "rows": 2})
        assert not destination.exists()
        P.publish_file_no_replace(partial, destination)

    assert not partial.exists()
    assert json.loads(destination.read_text()) == {"partial": False, "rows": 2}
    original = destination.read_bytes()
    competitor = tmp_path / "competitor.tmp"
    competitor.write_text("competitor\n")
    with pytest.raises(P.PublicationError, match="refusing to replace"):
        P.publish_file_no_replace(competitor, destination)
    assert destination.read_bytes() == original
    assert competitor.read_text() == "competitor\n"


def test_claim_contention_and_process_death_release_across_processes(tmp_path):
    destination = tmp_path / "process-result.json"
    identity = {"schema": "test.v1", "input_sha256": "c" * 64}
    program = f"""
import importlib.util, json, sys
from pathlib import Path
module_path = Path({str(_PATH)!r})
spec = importlib.util.spec_from_file_location("child_atomic_publication", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
destination = Path({str(destination)!r})
identity = {identity!r}
with module.exclusive_publication_claim(destination, identity=identity):
    print("READY", flush=True)
    sys.stdin.readline()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", program],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        ready, _, _ = select.select([child.stdout], [], [], 10)
        assert ready, "child did not acquire the claim within 10 seconds"
        assert child.stdout.readline().strip() == "READY"
        with pytest.raises(P.PublicationError, match="competing producer"):
            with P.exclusive_publication_claim(destination, identity=identity):
                pass
        # Simulate an ungraceful producer death: the kernel must release flock.
        child.kill()
        child.wait(timeout=10)
        with P.exclusive_publication_claim(destination, identity=identity):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
