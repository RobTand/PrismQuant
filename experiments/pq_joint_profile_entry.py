"""Run joint preparation/cost with its declared profiler inside the PB container."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiments.profile_command import run_profile


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    options, _ = parser.parse_known_args(args)
    raw = options.plan.read_bytes()
    if hashlib.sha256(raw).hexdigest() != options.plan_sha256:
        raise ValueError("profile launch plan checksum changed")
    plan = json.loads(raw)
    command = [sys.executable, "-m", "prismaquant.tessera_joint_aura", *args]
    if plan["profile_tool"] == "cprofile":
        return subprocess.run(command, check=False).returncode
    if plan["profile_tool"] != "py-spy":
        raise ValueError("unsupported joint profiler")
    target = Path("/tmp/pq-joint-sampling-tool")
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--quiet", "--target", str(target), "py-spy==0.4.2"], check=True)
    return run_profile(Path(plan["output_root"]) / options.command / "sampling", command,
                       profiler=str(target / "bin/py-spy"), rate=50)


if __name__ == "__main__":
    raise SystemExit(main())
