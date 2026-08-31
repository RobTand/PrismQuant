from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/numeric_execution_contract.py"
)
_SPEC = importlib.util.spec_from_file_location("numeric_execution_contract", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CONTRACT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTRACT)


def _current_environment(*, host="sparky", device="NVIDIA GB10"):
    return {
        "torch": "2.13.0+cu130",
        "python": "3.12.3",
        "host": host,
        "device": device,
        "triton": "3.7.1",
        "container_image": "sha256:" + "a" * 64,
    }


def _process_environment(*, host="sparky", image=None):
    return {
        "HULL_PHYSICAL_HOST": host,
        "HULL_CONTAINER_IMAGE": image or "sha256:" + "a" * 64,
    }


def _clean_git(monkeypatch):
    def fake_git(_root, *args, **_kwargs):
        if args == ("rev-parse", "HEAD"):
            return "b" * 40
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(_CONTRACT, "_git", fake_git)


def test_execution_identity_requires_checked_host_image_commit_and_clean_tree(
    tmp_path, monkeypatch,
):
    _clean_git(monkeypatch)
    monkeypatch.setattr(_CONTRACT.socket, "gethostname", lambda: "sparky")

    result = _CONTRACT.require_numeric_execution_environment(
        tmp_path,
        _current_environment(),
        _process_environment(),
        require_cuda=True,
    )

    assert result == {
        "schema": "trellis.numeric_execution.v1",
        "physical_host": "sparky",
        "container_image_digest": "sha256:" + "a" * 64,
        "repo_git_commit": "b" * 40,
        "repo_tree_clean": True,
        "python": "3.12.3",
        "torch": "2.13.0+cu130",
        "triton": "3.7.1",
        "device": "NVIDIA GB10",
    }


@pytest.mark.parametrize(
    ("current", "process", "live_host", "match"),
    [
        (_current_environment(), {}, "sparky", "HULL_PHYSICAL_HOST"),
        (
            _current_environment(host="container-id"),
            _process_environment(),
            "container-id",
            "host UTS namespace",
        ),
        (
            _current_environment(),
            _process_environment(image="latest"),
            "sparky",
            "immutable sha256 digest",
        ),
        (
            _current_environment(device=None),
            _process_environment(),
            "sparky",
            "CUDA device identity",
        ),
    ],
)
def test_execution_identity_refuses_unattested_runtime(
    tmp_path, monkeypatch, current, process, live_host, match,
):
    _clean_git(monkeypatch)
    monkeypatch.setattr(_CONTRACT.socket, "gethostname", lambda: live_host)
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match=match):
        _CONTRACT.require_numeric_execution_environment(
            tmp_path, current, process, require_cuda=True
        )


def test_execution_identity_refuses_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(_CONTRACT.socket, "gethostname", lambda: "sparky")

    def dirty_git(_root, *args, **_kwargs):
        if args == ("rev-parse", "HEAD"):
            return "b" * 40
        return " M research/driver.py"

    monkeypatch.setattr(_CONTRACT, "_git", dirty_git)
    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="clean repository worktree",
    ):
        _CONTRACT.require_numeric_execution_environment(
            tmp_path,
            _current_environment(),
            _process_environment(),
            require_cuda=True,
        )


def test_repo_commit_never_degrades_to_none(tmp_path, monkeypatch):
    monkeypatch.setattr(_CONTRACT, "_git", lambda *_args, **_kwargs: "unknown")
    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="not a full lowercase commit",
    ):
        _CONTRACT.require_repo_commit(tmp_path)
