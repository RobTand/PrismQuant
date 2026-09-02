"""Container image and mounted-PrismaQuant identity are resume inputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "container_runtime_identity.py"
IMAGE_REF = (
    "vllm-node@sha256:"
    "f7dad9260fea6f4207bd894acc9ebc034d91c599a70489a89ab1938a75db9c47"
)
IMAGE_ID = (
    "sha256:"
    "f7dad9260fea6f4207bd894acc9ebc034d91c599a70489a89ab1938a75db9c47"
)
COMMIT = "a" * 40


def _source_root(root: Path, *, marker: str = "reviewed") -> Path:
    package = root / "prismaquant"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f'MARKER = "{marker}"\n', encoding="utf-8"
    )
    (package / "data.json").write_text(
        json.dumps({"marker": marker}), encoding="utf-8"
    )
    return root


def _receipt(path: Path, *, image_ref: str = IMAGE_REF) -> Path:
    path.write_text(
        json.dumps(
            {
                "git_commit": COMMIT,
                # An extra per-lane block the tool no longer reads: the
                # --require-receipt-image cross-check that consumed it retired
                # 2026-09-02 with its lane (archive/gridbook_lane_2026-09-02/).
                # Kept as a shape the reader must tolerate, not require.
                "campaign": {"image": image_ref},
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(*args: object, env: dict[str, str] | None = None, cwd=None):
    return subprocess.run(
        [sys.executable, str(TOOL), *(str(arg) for arg in args)],
        cwd=cwd or REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_args(
    *, identity: Path, checkpoint: Path, source: Path, receipt: Path
) -> tuple[object, ...]:
    return (
        "write-or-verify",
        "--identity", identity,
        "--checkpoint-root", checkpoint,
        "--source-root", source,
        "--target", "dsv4",
        "--image-ref", IMAGE_REF,
        "--image-id", IMAGE_ID,
        "--git-commit", COMMIT,
        "--implementation-receipt", receipt,
    )


def test_runtime_identity_is_atomic_resume_boundary(tmp_path):
    source = _source_root(tmp_path / "source")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    identity = checkpoint / "container_runtime_identity.json"
    receipt = _receipt(tmp_path / "receipt.json")
    args = _write_args(
        identity=identity,
        checkpoint=checkpoint,
        source=source,
        receipt=receipt,
    )

    first = _run(*args)
    assert first.returncode == 0, first.stderr
    payload = json.loads(identity.read_text(encoding="utf-8"))
    assert payload["schema"] == "prismaquant.container_runtime_identity.v1"
    assert payload["image_ref"] == IMAGE_REF
    assert payload["image_id"] == IMAGE_ID
    assert payload["prismaquant_git_commit"] == COMMIT
    assert payload["implementation_receipt_sha256"] == hashlib.sha256(
        receipt.read_bytes()
    ).hexdigest()

    repeated = _run(*args)
    assert repeated.returncode == 0, repeated.stderr
    drifted = list(args)
    drifted[drifted.index("--image-id") + 1] = "sha256:" + "b" * 64
    drift = _run(*drifted)
    assert drift.returncode == 2
    assert "image_id" in drift.stderr


def test_source_sha256_cli_tracks_package_bytes_only(tmp_path):
    source = _source_root(tmp_path / "source")
    first = _run("source-sha256", "--source-root", source)
    assert first.returncode == 0, first.stderr
    assert len(first.stdout.strip()) == 64

    (source / "outside.txt").write_text("not package identity", encoding="utf-8")
    outside = _run("source-sha256", "--source-root", source)
    assert outside.returncode == 0, outside.stderr
    assert outside.stdout == first.stdout

    (source / "prismaquant" / "data.json").write_text(
        json.dumps({"marker": "changed"}), encoding="utf-8"
    )
    changed = _run("source-sha256", "--source-root", source)
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout != first.stdout


def test_runtime_identity_rejects_mutable_image_tag(tmp_path):
    source = _source_root(tmp_path / "source")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    identity = checkpoint / "container_runtime_identity.json"
    args = list(_write_args(
        identity=identity,
        checkpoint=checkpoint,
        source=source,
        receipt=_receipt(tmp_path / "receipt.json"),
    ))
    args[args.index("--image-ref") + 1] = "vllm-node:test"
    result = _run(*args)
    assert result.returncode == 2
    assert "image_ref is not immutable" in result.stderr
    assert not identity.exists()


def test_runtime_identity_never_adopts_legacy_checkpoints(tmp_path):
    source = _source_root(tmp_path / "source")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "old-unit.pt").write_bytes(b"unattested")
    identity = checkpoint / "container_runtime_identity.json"
    result = _run(
        *_write_args(
            identity=identity,
            checkpoint=checkpoint,
            source=source,
            receipt=_receipt(tmp_path / "receipt.json"),
        )
    )
    assert result.returncode == 2
    assert "predates container runtime identity" in result.stderr
    assert not identity.exists()


def test_in_container_check_refuses_stale_python_package(tmp_path):
    source = _source_root(tmp_path / "source")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    identity = checkpoint / "container_runtime_identity.json"
    created = _run(
        *_write_args(
            identity=identity,
            checkpoint=checkpoint,
            source=source,
            receipt=_receipt(tmp_path / "receipt.json"),
        )
    )
    assert created.returncode == 0, created.stderr

    verify_args = (
        "verify-mounted",
        "--identity", identity,
        "--expected-root", source,
        "--expected-image-ref", IMAGE_REF,
        "--expected-image-id", IMAGE_ID,
        "--expected-git-commit", COMMIT,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source)
    environment["PYTHONNOUSERSITE"] = "1"
    exact = _run(*verify_args, env=environment, cwd=tmp_path)
    assert exact.returncode == 0, exact.stderr

    (source / "prismaquant" / "data.json").write_text(
        json.dumps({"marker": "mutated-after-attestation"}),
        encoding="utf-8",
    )
    mutated = _run(*verify_args, env=environment, cwd=tmp_path)
    assert mutated.returncode == 2
    assert "package bytes differ" in mutated.stderr

    # Restore the attested source so the next assertion isolates Python import
    # precedence rather than the byte-integrity check above.
    (source / "prismaquant" / "data.json").write_text(
        json.dumps({"marker": "reviewed"}), encoding="utf-8"
    )
    stale = _source_root(tmp_path / "stale", marker="old-site-package")
    environment["PYTHONPATH"] = str(stale)
    refused = _run(*verify_args, env=environment, cwd=tmp_path)
    assert refused.returncode == 2
    assert "outside the reviewed mount" in refused.stderr
