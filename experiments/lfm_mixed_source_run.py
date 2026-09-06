"""Task-local code-input prelude for an admitted mixed LFM observation.

Uses the already published archive/manifest, following joint CPU qualification's
local extraction convention. There is no persistent source cache or placement.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.lfm_mixed_serving import require, sha, read, verify_encoder


def extract_source(archive, manifest_path, digest, destination):
    require(sha(manifest_path) == digest, "source archive manifest changed")
    manifest = read(manifest_path)
    require(archive.stat().st_size == manifest["archive_bytes"] and sha(archive) == manifest["archive_sha256"],
            "source archive bytes changed")
    require(not destination.exists(), "fresh task-local source destination required")
    destination.mkdir()
    with tarfile.open(archive) as stream:
        require(all(member.isdir() or member.isfile() for member in stream.getmembers()),
                "source archive contains nonregular entries")
        stream.extractall(destination, filter="data")
    binding = verify_encoder(destination, manifest_path, digest)
    return {**binding, "archive_sha256": manifest["archive_sha256"], "archive_bytes": manifest["archive_bytes"],
            "files": len(manifest["files"]), "local_source": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-archive", type=Path, required=True)
    parser.add_argument("--encoder-manifest", type=Path, required=True)
    parser.add_argument("--encoder-manifest-sha256", required=True)
    parser.add_argument("wrapper_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    require(os.environ.get("PRISMABUILD_CONTAINER_OWNER"), "source staging requires admitted PB action")
    rest = args.wrapper_arguments
    if rest[:1] == ["--"]:
        rest = rest[1:]
    require(rest and not any(arg.split("=", 1)[0] in
            {"--encoder", "--encoder-manifest", "--encoder-manifest-sha256"} for arg in rest),
            "wrapper must use the verified local encoder binding")
    with tempfile.TemporaryDirectory(prefix="lfm-mixed-source-") as local:
        source = Path(local) / "encoder"
        binding = extract_source(args.encoder_archive, args.encoder_manifest,
                                 args.encoder_manifest_sha256, source)
        print(json.dumps({"schema": "prismaquant.lfm-mixed-local-source.v1", **binding}), flush=True)
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("lfm_mixed_serving.py")),
            "--encoder", str(source), "--encoder-manifest", str(args.encoder_manifest.resolve()),
            "--encoder-manifest-sha256", args.encoder_manifest_sha256, *rest],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
