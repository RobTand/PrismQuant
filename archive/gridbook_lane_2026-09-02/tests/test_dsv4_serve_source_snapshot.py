"""The DSv4 native gate's two source identities.

The SERVED STACK still executes only from the artifact-build snapshot. The
host-side JUDGE runs at the live checkout's HEAD, because binding it to the
build commit too made a validator bug incurable for bytes already on disk --
the only remedy the gate admitted was rebuilding bytes that were never wrong
(2026-08-16). `judge_divergence` is what keeps the split honest; its own
refusals are covered in `test_runtime_snapshot_judge_split.py`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "serve_dsv4_cb_validate.sh"


def test_serve_driver_has_snapshot_reexec_and_repeated_closure_proofs():
    source = DRIVER.read_text(encoding="utf-8")
    assert 'git = (card.get("build") or {}).get("git") or {}' in source
    assert "shipcard build.git.dirty is not false" in source
    assert '"$REPO/tools/prismaquant_runtime_snapshot.py"' in source
    assert 'materialize --source-root "$REPO"' in source
    assert 'exec bash "$PQ_JUDGE_SNAPSHOT/scripts/serve_dsv4_cb_validate.sh"' in source
    assert source.count("verify_runtime_snapshot") >= 8
    assert "--expected-closure-sha256" in source
    # The CONTAINER gets the runtime snapshot, never the judge.
    assert '"$PQ_RUNTIME_SNAPSHOT:/repo:ro"' in source
    assert '"$REPO:/repo:ro"' not in source
    assert 'prismaquant_source_bootstrap.py" run-module' in source
    assert "python3 -m prismaquant" not in source
    assert "PYTHONPATH=/repo" not in source
    assert "/repo/tools/prismaquant_runtime_snapshot.py verify" in source
    assert "run-tool --source-root /repo" in source
    assert "serve-fingerprint write" in source
    assert "/repo/tools/serve_fingerprint.py write" not in source
    assert "export PQ_RUNTIME_PRISMAQUANT_ROOT=$PQ_JUDGE_SNAPSHOT" in source
    assert "export PYTHONSAFEPATH=1" in source
    assert "export PYTHONDONTWRITEBYTECODE=1" in source
    assert "export PYTHONNOUSERSITE=1" in source
    assert "unset PYTHONPATH" in source
    assert '"${PQ_RUNTIME_PRISMAQUANT_ROOT:-}" != "$REPO"' in source
    assert '"${PYTHONSAFEPATH:-}" != 1' in source
    assert '"${PYTHONDONTWRITEBYTECODE:-}" != 1' in source
    assert '"${PYTHONNOUSERSITE:-}" != 1' in source
    assert '-e "PYTHONDONTWRITEBYTECODE=1"' in source
    assert '-e "PYTHONNOUSERSITE=1"' in source
    assert '-n "${PYTHONPATH+x}"' in source
    assert source.index("verify_runtime_snapshot") < source.index(
        '. "$PQ_RUNTIME_SNAPSHOT/prismaquant/gridbook_runtime/'
        'gridbook_serving_runtime.sh"'
    )


def test_serve_driver_materializes_and_proves_both_identities():
    """The split's structure, pinned where it can be read off the driver.

    Two snapshots, both verified, the divergence claim re-proved at every
    checkpoint rather than trusted once at bootstrap, and the artifact's own
    identity left alone -- the judge running newer does not restamp what
    produced the bytes.
    """

    source = DRIVER.read_text(encoding="utf-8")
    assert 'pq_materialize_snapshot "$ARTIFACT_BUILD_COMMIT"' in source
    assert 'pq_materialize_snapshot "$REPO_HEAD"' in source
    assert "merge-base --is-ancestor" in source
    assert source.count("judge-divergence") >= 2
    assert "export PRISMAQUANT_IDENTITY_GIT_COMMIT=$ARTIFACT_BUILD_COMMIT" in source
    assert '--expected-commit "$PQ_JUDGE_COMMIT"' in source
    assert "judge_split.json" in source
    # The divergence proof must be inside the repeated checkpoint helper, not
    # only in the one-shot bootstrap above it.
    helper = source.index("verify_runtime_snapshot() {")
    assert source.index("judge-divergence", helper) < source.index("}", helper + 200)


def test_serve_driver_refuses_a_judge_older_than_the_artifact(tmp_path):
    """Forward only. A checkout that is not a descendant cannot judge."""

    model = tmp_path / "artifact"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "quant_config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture")
    build_commit = "a" * 40
    head = "d" * 40
    (model / "shipcard.json").write_text(json.dumps({
        "build": {"git": {"commit": build_commit, "dirty": False}},
    }), encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "case \" $* \" in\n"
        "  *\" rev-parse --verify HEAD^{commit} \"*) echo '" + head + "' ;;\n"
        "  *\" status --porcelain --untracked-files=all \"*) ;;\n"
        "  *\" merge-base --is-ancestor \"*) exit 1 ;;\n"
        "  *) exit 91 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    environment = dict(os.environ)
    environment.update({
        "MODEL": str(model),
        "PATH": f"{fake_bin}:{environment['PATH']}",
    })
    result = subprocess.run(
        ["bash", str(DRIVER), "eager"],
        cwd="/",
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert "is not a descendant of the artifact build commit" in result.stderr
    assert "docker" not in result.stderr.lower()


def test_serve_driver_refuses_dirty_bootstrap_before_snapshot_or_docker(tmp_path):
    model = tmp_path / "artifact"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "quant_config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture")
    commit = "a" * 40
    (model / "shipcard.json").write_text(json.dumps({
        "build": {"git": {"commit": commit, "dirty": False}},
    }), encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "case \" $* \" in\n"
        "  *\" rev-parse --verify HEAD^{commit} \"*) echo '" + commit + "' ;;\n"
        "  *\" status --porcelain --untracked-files=all \"*) echo ' M dirty' ;;\n"
        "  *) exit 91 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    environment = dict(os.environ)
    environment.update({
        "MODEL": str(model),
        "PATH": f"{fake_bin}:{environment['PATH']}",
    })
    result = subprocess.run(
        ["bash", str(DRIVER), "eager"],
        cwd="/",
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert "serve checkout must be a clean Git checkout" in result.stderr
    assert "docker" not in result.stderr.lower()


def test_inner_serve_driver_requires_strict_snapshot_bootstrap_environment(
    tmp_path,
):
    model = tmp_path / "artifact"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "quant_config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture")
    (model / "shipcard.json").write_text("{}", encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONSAFEPATH", None)
    environment.pop("PQ_RUNTIME_PRISMAQUANT_ROOT", None)
    environment.update({
        "MODEL": str(model),
        "PQ_SERVE_SNAPSHOT_REEXEC": "1",
        "PQ_RUNTIME_SNAPSHOT": str(REPO),
        "ARTIFACT_BUILD_COMMIT": "a" * 40,
        "PQ_RUNTIME_TREE": "b" * 40,
        "PQ_RUNTIME_CLOSURE_SHA256": "c" * 64,
    })
    result = subprocess.run(
        ["bash", str(DRIVER), "graph"],
        cwd="/",
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert "not executing from its attested snapshot" in result.stderr
    assert "docker" not in result.stderr.lower()
