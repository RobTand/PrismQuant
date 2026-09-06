"""PB-admitted host wrapper for the mixed-LFM native ABI preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    raw = args.source_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.source_sha256:
        raise ValueError("Tessera source manifest changed")
    source = json.loads(raw)
    def verify_source():
        actual = {str(p.relative_to(args.source)) for p in args.source.rglob("*") if p.is_file()}
        if actual != set(source["files"]):
            raise ValueError("Tessera source population changed")
        for name, expected in source["files"].items():
            path = args.source / name
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Tessera source changed: {name}")
    verify_source()
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image]))[0]
    if args.image not in image.get("RepoDigests", []):
        raise ValueError("local daemon did not resolve the declared serving digest")
    owner = uuid.uuid4().hex
    name = f"lfm-mixed-native-{owner[:12]}"
    cidfile = args.out / "container.cid"
    record = {"schema": "prismaquant.lfm_mixed_native_host.v1", "source_manifest_sha256": args.source_sha256,
              "tessera_commit": source["commit"], "image_reference": args.image, "image_id": image["Id"],
              "source_snapshot": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "container_name": name,
              "started_unix": time.time(), "telemetry_errors": [], "cleanup_safe": False}
    stopped = threading.Event()
    def telemetry():
        with (args.out / "netdata.jsonl").open("w") as stream:
            while not stopped.is_set():
                for host in ("sparky", "sparklina"):
                    try:
                        with urllib.request.urlopen(f"http://{host}:19999/api/v1/allmetrics?format=json", timeout=5) as response:
                            metrics = json.load(response)
                        stream.write(json.dumps({"host": host, "time": time.time(), "metrics": metrics}) + "\n")
                        stream.flush()
                    except Exception as exc:
                        record["telemetry_errors"].append({"host": host, "error": repr(exc)})
                stopped.wait(10)
    monitor = threading.Thread(target=telemetry, daemon=True)
    monitor.start()
    command = ["docker", "run", "--rm", "--pull=never", "--gpus", "all", "--name", name,
               "--cidfile", str(cidfile), "--label", f"prismaquant.mixed-native={owner}",
               "--memory=16g", "--memory-swap=16g", "--cpus=4", "--ipc=host",
               "-v", f"{repo}:/pq:ro", "-v", f"{args.source}:/tessera:ro",
               "-v", f"{args.model}:/model:ro", "-v", f"{args.out}:/out", "-w", "/pq",
               "-e", "PYTHONPATH=/tessera/src:/pq", "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
               "--entrypoint", "python3", args.image, "experiments/lfm_mixed_native_preflight.py",
               "--model", "/model", "--out", "/out/probe", "--image", args.image,
               "--tessera-commit", source["commit"]]
    record["argv"] = command
    try:
        with (args.out / "run.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        record["returncode"] = result.returncode
        result.check_returncode()
        verify_source()
    finally:
        stopped.set()
        monitor.join(timeout=15)
        if cidfile.exists():
            cid = cidfile.read_text().strip()
            inspected = subprocess.run(["docker", "inspect", cid], capture_output=True, text=True)
            if inspected.returncode == 0:
                container = json.loads(inspected.stdout)[0]
                if container["Id"] != cid or container["Config"]["Labels"].get("prismaquant.mixed-native") != owner:
                    raise RuntimeError("refusing cleanup of a container with different ownership")
                subprocess.run(["docker", "rm", "-f", cid], check=True, capture_output=True)
            record["container_id"] = cid
            record["cleanup_safe"] = not subprocess.check_output(
                ["docker", "ps", "-aq", "--filter", f"name=^/{name}$"], text=True).strip()
        record["finished_unix"] = time.time()
        (args.out / "host.json").write_text(json.dumps(record, indent=2) + "\n")
    if not record["cleanup_safe"] or record["telemetry_errors"]:
        raise RuntimeError("preflight cleanup or host telemetry is incomplete")
    print(json.dumps({"passed": True, "cleanup_safe": True, "out": str(args.out)}))


if __name__ == "__main__":
    main()
