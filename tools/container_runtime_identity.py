#!/usr/bin/env python3
"""Fail-closed identity for containerized PrismaQuant producer runs.

The container image supplies CUDA/PyTorch/Transformers, while PrismaQuant is
mounted from the reviewed checkout.  Both inputs affect rendered bytes and
must therefore be stable across an interrupted campaign.  This helper writes
one atomic identity before the first container starts and verifies, inside the
container, that Python will import the exact mounted package bytes.

It is intentionally stdlib-only and is invoked by path rather than through
``python -m prismaquant``: checking which PrismaQuant would be imported must
happen before importing PrismaQuant itself.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


RUNTIME_IDENTITY_SCHEMA = "prismaquant.container_runtime_identity.v1"
_IDENTITY_KEYS = {
    "schema",
    "target",
    "image_ref",
    "image_id",
    "prismaquant_git_commit",
    "prismaquant_source_sha256",
    "implementation_receipt_sha256",
}
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
_IMAGE_REF_RE = re.compile(
    r"(?:[^@\s]+@sha256:[0-9a-f]{64}|sha256:[0-9a-f]{64})"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class RuntimeIdentityError(ValueError):
    """The producer runtime cannot be identified exactly."""


def _reject_duplicate_members(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeIdentityError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, where: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_members,
        )
    except RuntimeIdentityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeIdentityError(f"{where} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeIdentityError(f"{where} must be a JSON object")
    return payload


def prismaquant_source_sha256(package_root: Path) -> str:
    """Hash the same durable package scope used by CB pair identity."""

    root = package_root.resolve(strict=True)
    if not root.is_dir() or root.name != "prismaquant":
        raise RuntimeIdentityError(
            f"PrismaQuant package root must name a prismaquant directory: {root}"
        )
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    if not paths:
        raise RuntimeIdentityError(f"no package inputs found under {root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise RuntimeIdentityError(
                f"cannot hash PrismaQuant package input {path}"
            ) from exc
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _validate_identity(payload: Mapping[str, Any], *, where: str) -> dict[str, str]:
    if set(payload) != _IDENTITY_KEYS:
        raise RuntimeIdentityError(
            f"{where} expected exactly {sorted(_IDENTITY_KEYS)}, "
            f"got {sorted(payload)}"
        )
    values = {str(key): value for key, value in payload.items()}
    if values["schema"] != RUNTIME_IDENTITY_SCHEMA:
        raise RuntimeIdentityError(
            f"{where} has unsupported schema {values['schema']!r}"
        )
    target = values["target"]
    if target not in {"dense", "dsv4"}:
        raise RuntimeIdentityError(f"{where} has invalid target {target!r}")
    image_ref = values["image_ref"]
    image_id = values["image_id"]
    commit = values["prismaquant_git_commit"]
    source_sha = values["prismaquant_source_sha256"]
    receipt_sha = values["implementation_receipt_sha256"]
    if not isinstance(image_ref, str) or _IMAGE_REF_RE.fullmatch(image_ref) is None:
        raise RuntimeIdentityError(f"{where} image_ref is not immutable")
    if not isinstance(image_id, str) or _IMAGE_ID_RE.fullmatch(image_id) is None:
        raise RuntimeIdentityError(f"{where} image_id is not a full Docker ID")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeIdentityError(f"{where} PrismaQuant commit is invalid")
    for name, value in (
        ("prismaquant_source_sha256", source_sha),
        ("implementation_receipt_sha256", receipt_sha),
    ):
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise RuntimeIdentityError(f"{where} {name} is invalid")
    return {key: str(value) for key, value in values.items()}


def _receipt_sha256(
    receipt_path: Path,
    *,
    expected_commit: str,
) -> str:
    try:
        data = receipt_path.read_bytes()
    except OSError as exc:
        raise RuntimeIdentityError(
            f"implementation receipt is unreadable: {receipt_path}"
        ) from exc
    receipt = _load_json_object(receipt_path, where="implementation receipt")
    if receipt.get("git_commit") != expected_commit:
        raise RuntimeIdentityError(
            "implementation receipt commit does not match the mounted "
            f"PrismaQuant commit {expected_commit}"
        )
    # --require-receipt-image also pinned the receipt's campaign image, but it
    # read that image from a `gridbook_runtime` block (retired 2026-09-02, see
    # archive/gridbook_lane_2026-09-02/) that only one lane wrote, and only
    # that lane's archived drivers ever passed the flag. The image tag
    # itself is still fatal on its own: _validate_identity requires an
    # immutable digest-pinned image_ref, and verify_mounted_runtime compares it.
    return hashlib.sha256(data).hexdigest()


def _identity_diff(
    expected: Mapping[str, str], observed: Mapping[str, str]
) -> list[str]:
    return sorted(
        key
        for key in _IDENTITY_KEYS
        if expected.get(key) != observed.get(key)
    )


def write_or_verify_identity(
    *,
    identity_path: Path,
    checkpoint_root: Path,
    source_root: Path,
    target: str,
    image_ref: str,
    image_id: str,
    git_commit: str,
    implementation_receipt: Path,
) -> dict[str, str]:
    package_root = source_root.resolve(strict=True) / "prismaquant"
    # Validate the transport identity before consulting its receipt. This
    # makes a mutable image tag an independently fatal input rather than an
    # incidental receipt mismatch.
    payload = _validate_identity(
        {
            "schema": RUNTIME_IDENTITY_SCHEMA,
            "target": target,
            "image_ref": image_ref,
            "image_id": image_id,
            "prismaquant_git_commit": git_commit,
            "prismaquant_source_sha256": "0" * 64,
            "implementation_receipt_sha256": "0" * 64,
        },
        where="requested container runtime identity",
    )
    payload["prismaquant_source_sha256"] = prismaquant_source_sha256(package_root)
    payload["implementation_receipt_sha256"] = _receipt_sha256(
        implementation_receipt,
        expected_commit=git_commit,
    )

    checkpoint = checkpoint_root.resolve(strict=True)
    destination = identity_path.resolve(strict=False)
    if checkpoint not in destination.parents:
        raise RuntimeIdentityError(
            f"runtime identity must live under checkpoint root {checkpoint}"
        )
    if destination.exists():
        observed = _validate_identity(
            _load_json_object(destination, where="stored container runtime identity"),
            where="stored container runtime identity",
        )
        differing = _identity_diff(payload, observed)
        if differing:
            raise RuntimeIdentityError(
                "container runtime differs from the resumable campaign in: "
                + ", ".join(differing)
            )
        return payload

    # Never retroactively bless checkpoints made before image/source identity
    # existed. A fresh work tree can acquire its identity; a legacy journal
    # needs an explicit one-time audit/migration rather than a guess.
    legacy = sorted(
        path for path in checkpoint.rglob("*")
        if path.is_file() and path.resolve(strict=False) != destination
    )
    if legacy:
        preview = ", ".join(str(path) for path in legacy[:3])
        raise RuntimeIdentityError(
            "checkpoint tree predates container runtime identity; refusing "
            f"to adopt existing files ({preview})"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def verify_mounted_runtime(
    *,
    identity_path: Path,
    expected_root: Path,
    expected_image_ref: str,
    expected_image_id: str,
    expected_git_commit: str,
) -> dict[str, str]:
    payload = _validate_identity(
        _load_json_object(identity_path, where="container runtime identity"),
        where="container runtime identity",
    )
    transported = {
        "image_ref": expected_image_ref,
        "image_id": expected_image_id,
        "prismaquant_git_commit": expected_git_commit,
    }
    differing = sorted(
        key for key, value in transported.items() if payload.get(key) != value
    )
    if differing:
        raise RuntimeIdentityError(
            "container runtime transport differs from host identity in: "
            + ", ".join(differing)
        )

    root = expected_root.resolve(strict=True)
    expected_package = root / "prismaquant"
    expected_origin = (expected_package / "__init__.py").resolve(strict=True)
    spec = importlib.util.find_spec("prismaquant")
    if spec is None or spec.origin is None:
        raise RuntimeIdentityError("Python cannot resolve PrismaQuant")
    try:
        observed_origin = Path(spec.origin).resolve(strict=True)
    except OSError as exc:
        raise RuntimeIdentityError(
            f"resolved PrismaQuant origin is unreadable: {spec.origin}"
        ) from exc
    if observed_origin != expected_origin:
        raise RuntimeIdentityError(
            "Python resolves PrismaQuant outside the reviewed mount: "
            f"{observed_origin} != {expected_origin}"
        )
    source_sha = prismaquant_source_sha256(expected_package)
    if source_sha != payload["prismaquant_source_sha256"]:
        raise RuntimeIdentityError(
            "mounted PrismaQuant package bytes differ from the host-attested "
            "source identity"
        )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write-or-verify")
    write.add_argument("--identity", type=Path, required=True)
    write.add_argument("--checkpoint-root", type=Path, required=True)
    write.add_argument("--source-root", type=Path, required=True)
    write.add_argument("--target", choices=("dense", "dsv4"), required=True)
    write.add_argument("--image-ref", required=True)
    write.add_argument("--image-id", required=True)
    write.add_argument("--git-commit", required=True)
    write.add_argument("--implementation-receipt", type=Path, required=True)

    verify = subparsers.add_parser("verify-mounted")
    verify.add_argument("--identity", type=Path, required=True)
    verify.add_argument("--expected-root", type=Path, required=True)
    verify.add_argument("--expected-image-ref", required=True)
    verify.add_argument("--expected-image-id", required=True)
    verify.add_argument("--expected-git-commit", required=True)

    source_sha = subparsers.add_parser(
        "source-sha256",
        help="print the canonical hash of one mounted PrismaQuant package",
    )
    source_sha.add_argument("--source-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "source-sha256":
            digest = prismaquant_source_sha256(
                args.source_root.resolve(strict=True) / "prismaquant"
            )
            print(digest)
            return 0
        if args.command == "write-or-verify":
            payload = write_or_verify_identity(
                identity_path=args.identity,
                checkpoint_root=args.checkpoint_root,
                source_root=args.source_root,
                target=args.target,
                image_ref=args.image_ref,
                image_id=args.image_id,
                git_commit=args.git_commit,
                implementation_receipt=args.implementation_receipt,
            )
        else:
            payload = verify_mounted_runtime(
                identity_path=args.identity,
                expected_root=args.expected_root,
                expected_image_ref=args.expected_image_ref,
                expected_image_id=args.expected_image_id,
                expected_git_commit=args.expected_git_commit,
            )
    except RuntimeIdentityError as exc:
        print(f"container-runtime: ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
