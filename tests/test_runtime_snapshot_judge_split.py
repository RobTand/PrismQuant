"""The serve gate's judge/runtime split, and the proof that keeps it honest.

Until 2026-08-16 the gate bound the *judge* to the artifact's build commit and
then materialized and ran the snapshot at that commit, which made a validator
bug structurally incurable: the only remedy the gate admitted for a wrong
verdict was rebuilding bytes that were never wrong. The container still runs the
artifact's own build commit -- nothing about the serve is relaxed -- but the
host-side verdict now runs at HEAD, and `judge_divergence` is the obligation
that makes that safe: every closure path that differs between the two revisions
must be judge-only, or the gate refuses and re-export is the honest answer.

These tests exist because the interesting direction is the refusal. A proof that
only ever passes is the same defect as a gate no artifact can pass.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "prismaquant_runtime_snapshot.py"
_spec = importlib.util.spec_from_file_location("_pq_runtime_snapshot_tool", _TOOL)
assert _spec is not None and _spec.loader is not None
snapshot_tool = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = snapshot_tool
_spec.loader.exec_module(snapshot_tool)


_COMMIT_A = "a" * 40
_COMMIT_B = "b" * 40
_TREE = "c" * 40


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_snapshot(root: Path, files: dict[str, str]) -> Path:
    """A snapshot is read through its manifest, so the manifest is the fixture."""

    root.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "path": path,
            "type": "file",
            "bytes": len(content),
            "executable": False,
            "sha256": _digest(content),
        }
        for path, content in sorted(files.items())
    ]
    payload = {
        "schema": snapshot_tool.SCHEMA,
        "commit": _COMMIT_A,
        "tree": _TREE,
        "closure_sha256": _digest(json.dumps(entries, sort_keys=True)),
        "entry_count": len(entries),
        "entries": entries,
    }
    (root / snapshot_tool.MANIFEST).write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return root


#: A minimal stand-in for the serve path: what the container actually executes.
_RUNTIME_FILES = {
    # Two lane-owned stand-ins left this table on 2026-09-02 with the Gridbook
    # lane (archive/gridbook_lane_2026-09-02/): the runtime entry
    # `prismaquant/gridbook_runtime/gridbook_serving_runtime.sh` and the
    # judge-only entry `prismaquant/validate_cb_endpoint.py`. They were fixture
    # strings, not subjects -- what these tests prove is the *classification*,
    # so each is replaced by a live path of the same class rather than dropped.
    "prismaquant/model_profiles/deepseek_v4.py": "profile v1",
    "prismaquant/export_native_compressed.py": "runtime v1",
    "tools/prismaquant_source_bootstrap.py": "bootstrap v1",
    "prismaquant/artifact_completeness.py": "completeness v1",
    "tests/test_something.py": "test v1",
    "docs/ARCHITECTURE.md": "doc v1",
}


def test_an_unchanged_judge_diverges_nowhere(tmp_path: Path) -> None:
    runtime = _write_snapshot(tmp_path / "runtime", _RUNTIME_FILES)
    judge = _write_snapshot(tmp_path / "judge", _RUNTIME_FILES)

    report = snapshot_tool.judge_divergence(runtime, judge)

    assert report["divergent_count"] == 0
    assert report["divergent_paths"] == []


def test_a_judge_only_change_is_the_case_the_split_exists_for(tmp_path: Path) -> None:
    """Exactly the shape of 2027c60: the gate moved, the served stack did not."""

    runtime = _write_snapshot(tmp_path / "runtime", _RUNTIME_FILES)
    judge = _write_snapshot(tmp_path / "judge", {
        **_RUNTIME_FILES,
        "prismaquant/artifact_completeness.py": "completeness v2",
        "tests/test_something.py": "test v2",
        "tests/test_added.py": "brand new",
        "docs/ARCHITECTURE.md": "doc v2",
    })

    report = snapshot_tool.judge_divergence(runtime, judge)

    # Was 5 until 2026-09-02, when `prismaquant/validate_cb_endpoint.py` --
    # one of the two named judge-only files -- retired with the Gridbook lane.
    assert report["divergent_count"] == 4
    assert "prismaquant/artifact_completeness.py" in report["divergent_paths"]
    assert "tests/test_added.py" in report["divergent_paths"]


@pytest.mark.parametrize(
    "path",
    [
        "prismaquant/model_profiles/deepseek_v4.py",
        "prismaquant/export_native_compressed.py",
        "tools/prismaquant_source_bootstrap.py",
    ],
)
def test_a_serve_path_change_refuses_the_split(tmp_path: Path, path: str) -> None:
    """The judge may run newer. It may not judge a stack it also changed.

    A newer judge that also moved the serve path could be expecting something
    the build-commit runtime does not produce. That is not a validator fix, and
    the honest answer there is a re-export.
    """

    runtime = _write_snapshot(tmp_path / "runtime", _RUNTIME_FILES)
    judge = _write_snapshot(tmp_path / "judge", {**_RUNTIME_FILES, path: "moved"})

    with pytest.raises(snapshot_tool.SnapshotError, match=path):
        snapshot_tool.judge_divergence(runtime, judge)


def test_adding_a_serve_path_file_refuses(tmp_path: Path) -> None:
    runtime = _write_snapshot(tmp_path / "runtime", _RUNTIME_FILES)
    judge = _write_snapshot(tmp_path / "judge", {
        **_RUNTIME_FILES, "prismaquant/new_kernel.py": "new",
    })

    with pytest.raises(snapshot_tool.SnapshotError, match="prismaquant/new_kernel.py"):
        snapshot_tool.judge_divergence(runtime, judge)


def test_deleting_a_serve_path_file_refuses(tmp_path: Path) -> None:
    """Absence is a divergence too -- comparing only shared paths would miss it."""

    remaining = dict(_RUNTIME_FILES)
    del remaining["prismaquant/model_profiles/deepseek_v4.py"]
    runtime = _write_snapshot(tmp_path / "runtime", _RUNTIME_FILES)
    judge = _write_snapshot(tmp_path / "judge", remaining)

    with pytest.raises(snapshot_tool.SnapshotError, match="deepseek_v4.py"):
        snapshot_tool.judge_divergence(runtime, judge)


def test_the_subtree_allowance_does_not_leak_to_a_sibling_prefix(
    tmp_path: Path,
) -> None:
    """`tests/` must not silently admit `testsuite/`, nor `docs/` `docs_build/`."""

    runtime = _write_snapshot(tmp_path / "runtime", _RUNTIME_FILES)
    judge = _write_snapshot(tmp_path / "judge", {
        **_RUNTIME_FILES, "testsuite/harness.py": "not a test dir",
    })

    with pytest.raises(snapshot_tool.SnapshotError, match="testsuite/harness.py"):
        snapshot_tool.judge_divergence(runtime, judge)


def test_a_near_miss_on_a_named_judge_file_refuses(tmp_path: Path) -> None:
    """The non-subtree entries are exact paths, not prefixes."""

    runtime = _write_snapshot(tmp_path / "runtime", _RUNTIME_FILES)
    judge = _write_snapshot(tmp_path / "judge", {
        **_RUNTIME_FILES, "prismaquant/validate_cb_endpoint_helpers.py": "sneaky",
    })

    with pytest.raises(snapshot_tool.SnapshotError, match="validate_cb_endpoint_helpers"):
        snapshot_tool.judge_divergence(runtime, judge)


def test_the_judge_only_list_stays_short_and_named(tmp_path: Path) -> None:
    """Growing this list is a contract decision, so make growing it visible.

    Not a tautology check on the values: it asserts the *shape* the split
    depends on -- nothing in the serve path may be waved through by a broad
    subtree, so every non-doc/test entry must be one exact file.
    """

    subtrees = [p for p in snapshot_tool.JUDGE_ONLY_PATHS if p.endswith("/")]
    exact = [p for p in snapshot_tool.JUDGE_ONLY_PATHS if not p.endswith("/")]

    assert sorted(subtrees) == ["docs/", "tests/"]
    assert all(p.endswith((".py", ".sh")) for p in exact)
    assert len(exact) <= 6, "judge-only code files should be few and justified"


# `test_the_snapshot_tool_is_the_one_entry_the_container_also_runs` was deleted
# on 2026-09-02. It read `scripts/serve_dsv4_cb_validate.sh` and proved that
# the launcher exercises the RUNTIME snapshot's copy of
# `tools/prismaquant_runtime_snapshot.py` host-side, with the identical
# `verify` CLI, before any container starts -- the stated reason that entry is
# allowed to differ between judge and runtime at all. That launcher went to
# archive/gridbook_lane_2026-09-02/scripts/ with the Gridbook lane.
#
# The allowlist entry is KEPT and the argument for it is unchanged, but no
# live launcher demonstrates it any more: it is now a justification with no
# executable witness. Whoever wires the next serve launcher must restore this
# test against it. Recorded as debt D34.


def test_the_two_gate_files_of_this_very_change_are_allowed(tmp_path: Path) -> None:
    """The guard caught this commit's own launcher edit before it was justified.

    Keeping the case: a change to the snapshot tool is judge-side, but
    `scripts/` is NOT a blanket subtree allowance -- a producer script under it
    still refuses.
    """

    runtime = _write_snapshot(tmp_path / "runtime", {
        **_RUNTIME_FILES,
        "scripts/export_something.sh": "producer v1",
        "tools/prismaquant_runtime_snapshot.py": "snapshot v1",
    })
    judge = _write_snapshot(tmp_path / "judge", {
        **_RUNTIME_FILES,
        "scripts/export_something.sh": "producer v1",
        "tools/prismaquant_runtime_snapshot.py": "snapshot v2",
    })

    # Was 2 until 2026-09-02: the serve launcher
    # `scripts/serve_dsv4_cb_validate.sh` was the OTHER of the "two gate files"
    # and left JUDGE_ONLY_PATHS with the Gridbook lane
    # (archive/gridbook_lane_2026-09-02/). One is left, and the half of this
    # test that matters is the half below.
    report = snapshot_tool.judge_divergence(runtime, judge)
    assert report["divergent_count"] == 1

    moved_producer = _write_snapshot(tmp_path / "judge2", {
        **_RUNTIME_FILES,
        "scripts/export_something.sh": "producer v2",
        "tools/prismaquant_runtime_snapshot.py": "snapshot v1",
    })
    with pytest.raises(snapshot_tool.SnapshotError, match="export_something.sh"):
        snapshot_tool.judge_divergence(runtime, moved_producer)
