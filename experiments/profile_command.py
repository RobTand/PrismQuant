"""Sample an admitted command without deterministic per-call profiling.

The child writes its own exit receipt: a successful profiler process alone
does not establish that the observed command succeeded. Run through PB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_child(output, command):
    started = time.time()
    result = {"schema": "prismaquant.profiled_command_exit.v1", "command": command,
              "started_unix": started, "returncode": None}
    try:
        child = subprocess.run(command, check=False)
        result["returncode"] = child.returncode
        return child.returncode
    finally:
        result["finished_unix"] = time.time()
        write_json(output / "child-exit.json", result)


def run_profile(output, command, *, profiler, rate):
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    raw = output / "stacks.raw"
    argv = [profiler, "record", "--subprocesses", "--rate", str(rate),
            "--format", "raw", "--output", str(raw), "--", sys.executable,
            str(Path(__file__).resolve()), "--child", "--output", str(output), "--", *command]
    with (output / "sampler.log").open("x") as log:
        sampled = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
    result = {"schema": "prismaquant.sampled_command.v1", "command": command,
              "profiler_command": argv, "profiler_returncode": sampled.returncode,
              "started_unix": started, "finished_unix": time.time(),
              "passed": False, "child": None, "artifacts": {}}
    receipt = output / "child-exit.json"
    if receipt.is_file():
        result["child"] = json.loads(receipt.read_text())
    for path in (receipt, raw, output / "sampler.log"):
        if path.is_file():
            blob = path.read_bytes()
            result["artifacts"][path.name] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    child = result["child"]
    result["passed"] = (sampled.returncode == 0 and isinstance(child, dict)
                        and child.get("command") == command and type(child.get("returncode")) is int
                        and child["returncode"] == 0 and raw.is_file() and raw.stat().st_size > 0)
    write_json(output / "profile-result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["passed"]:
        return 0
    if isinstance(child, dict) and type(child.get("returncode")) is int and child["returncode"]:
        return child["returncode"] if child["returncode"] > 0 else 128 - child["returncode"]
    return sampled.returncode or 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiler", default="py-spy")
    parser.add_argument("--rate", type=int, default=50)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= args.rate <= 1000:
        parser.error("a command and a sampling rate in [1,1000] are required")
    output = args.output.resolve()
    if args.child:
        return run_child(output, command)
    return run_profile(output, command, profiler=args.profiler, rate=args.rate)


if __name__ == "__main__":
    raise SystemExit(main())
