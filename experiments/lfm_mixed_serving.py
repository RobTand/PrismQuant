"""Opt-in #253: sealed mixed artifact → actual routes → strict gate → paired smoke.

This is one bounded admitted validation action, not an exporter or scheduler.
The strict result remains separate from the observational sanity result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.pq183_lfm_bound import IMAGE, sha, require, read, write
from experiments.pq87_physical_ab import cleanup_container, _http
from experiments.pq87_paired_validation import require_container_name_available

# tessera master 5ac9b72 (#378): a dense owner's roles are the runtime's attested
# output partitions, so the 18 quantized ShortConv in_proj tensors are emitted as
# three row-window roles each (serving_gate.geometry.row_sliced_modules).
# tessera master d19e3ad (#381/#382): the census roster counts a row-sliced
# owner's source tensor once, not once per role. experiments/ only, so the
# encoder fixture id of an artifact encoded at 5ac9b72 is unchanged and such
# an artifact serves under this pin; the seal records both commits.
ENCODER = "d19e3ad9aec3648f42a76f4d6bb88aa79e288144"
CALIBRATION_SEAL = "d302740baa39a484135a5de73abfcf9d5c3ec91eb419cb59a3d4af94c2704952"
SCALES = "08bce65811467a0620f1140c17d4a01a8198b49fa18c32b9603faefb5bc5bbe6"
PROMPTS = "1d280c010c23d156493e89d015afaad04cf94f434b3f4f60e6e797a0adb88375"
LABEL = "prismaquant.mixed-serving"
LIMITS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
          "MAX_JOBS": "4", "CMAKE_BUILD_PARALLEL_LEVEL": "4", "NINJAFLAGS": "-j4", "MAKEFLAGS": "-j4"}


def verify_encoder(root, manifest_path, digest):
    require(sha(manifest_path) == digest, "encoder manifest changed")
    manifest = read(manifest_path)
    require(manifest.get("commit") == ENCODER, "wrong encoder source commit")
    actual = {}
    for path in root.rglob("*"):
        require(not path.is_symlink(), "encoder source symlink")
        if path.is_file():
            actual[str(path.relative_to(root))] = sha(path)
    require(actual == manifest["files"], "encoder source closure changed")
    return {"commit": ENCODER, "manifest_sha256": digest}


def verify_assembly(artifact, result_path, digest):
    require(sha(result_path) == digest, "assembly result changed")
    records = [line.removeprefix("PB_TESSERA_RESULT=")
               for line in result_path.read_text().splitlines() if line.startswith("PB_TESSERA_RESULT=")]
    require(len(records) == 1, "assembly stdout requires exactly one PB completion record")
    result = json.loads(records[0])
    require(read(artifact / "pb-result.json") == result, "artifact differs from actual PB result")
    require(result.get("schema") == "prismabuild.tessera-model.v1" and "index" in result
            and result["index"] is None and bool(result.get("files")), "not an assembled PB result")
    actual = {}
    for path in artifact.iterdir():
        require(path.is_file() and not path.is_symlink(), "unsafe assembled artifact entry")
        if path.name != "pb-result.json":
            actual[path.name] = sha(path)
    require(actual == result["files"], "assembled output bytes/population changed")
    return result


def validate_handoff(plan, manifest, source_identity, calibration):
    identity = manifest["export_identity"]
    require("export_partition" not in manifest and bool(manifest.get("merged_from")), "not assembled export")
    require(identity["source"] == source_identity, "teacher source differs from export authority")
    require(identity["runtime_image"] == IMAGE, "wrong actual encoder image")
    options = identity["options"]
    require(options["plan"] == plan, "common export plan changed")
    require(options["hessian_sha256"] is None and options["input_scales_sha256"] == SCALES,
            "weights-only calibration binding changed")
    require(calibration["mode"] == "calibrate" and calibration["weights_only_export"] is True
            and calibration["hessian"] is None, "not a calibrated weights-only preparation")
    require(Counter((v["grid"], v["q256"]) for v in plan.values()) ==
            Counter({("E4M3", 1024): 22, ("E2M1x2", 896): 6, ("BF16", 1792): 60}),
            "mixed family population changed")
    require(manifest["totals"]["modules"] == 74 and manifest["totals"]["units"] == 2214,
            "incomplete full-body export")
    require(bool(identity.get("encoder_fixture_id")) and bool(identity.get("code_sha256")),
            "missing encoder numerical/source identity")
    return identity


def admission():
    """How this observation was admitted, stamped into ``host-status.json``.

    Batch work is admitted by PrismaBuild and carries ``PRISMABUILD_CONTAINER_OWNER``.
    Serving vLLM is exempt from PrismaBuild (Rob, 2026-09-06): a direct run
    declares itself with ``PRISMAQUANT_DIRECT_SERVE=<reason>`` and is stamped
    ``direct`` so it can never be read as an admitted action. Exactly one of
    the two must be declared; an undeclared run is refused as before.
    """
    action = os.environ.get("PRISMABUILD_CONTAINER_OWNER")
    direct = os.environ.get("PRISMAQUANT_DIRECT_SERVE")
    require(bool(action) != bool(direct),
            "declare exactly one admission: a PrismaBuild action or PRISMAQUANT_DIRECT_SERVE=<reason>")
    if action:
        return {"mode": "prismabuild", "action": action}
    return {"mode": "direct", "reason": direct, "policy": "vLLM serving is exempt from PrismaBuild (2026-09-06)"}


def verify_owned(container, cid, owner, cpus):
    require(container["Id"] == cid and container["Config"]["Labels"].get(LABEL) == owner,
            "refusing container with different ownership")
    config = container["HostConfig"]
    require(config["Memory"] == 64 * 2**30 and config["MemorySwap"] == 64 * 2**30,
            "container memory bounds changed")
    actual_cpus = set()
    for part in config.get("CpusetCpus", "").split(","):
        bounds = part.split("-")
        require(len(bounds) in (1, 2) and all(b.isdigit() for b in bounds), "invalid container affinity")
        actual_cpus.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    require(actual_cpus == set(cpus), "container affinity changed")
    env = dict(item.split("=", 1) for item in container["Config"]["Env"] if "=" in item)
    require(all(env.get(k) == v for k, v in LIMITS.items()), "container native/build bounds changed")


def public_container(inspected):
    """Retain runtime/resource facts, never arbitrary daemon Env or PB secrets."""
    env = dict(item.split("=", 1) for item in inspected["Config"]["Env"] if "=" in item)
    return {"Id": inspected["Id"], "Image": inspected.get("Image"),
            "owner": inspected["Config"]["Labels"].get(LABEL),
            "native_environment": {key: env.get(key) for key in LIMITS},
            "resources": {key: inspected["HostConfig"].get(key)
                          for key in ("Memory", "MemorySwap", "CpusetCpus", "NanoCpus")},
            "running": inspected.get("State", {}).get("Running")}


def cleanup_owned(name, cidpath, owner):
    """Only a captured immutable ID with our label may reach the cleanup helper."""
    if not cidpath.exists():
        require_container_name_available(name)
        return {"safe": True, "never_created": True}
    cid = cidpath.read_text().strip()
    require(re.fullmatch(r"[0-9a-f]{64}", cid), "invalid owned container ID")
    got = subprocess.run(["docker", "inspect", cid], capture_output=True, text=True, timeout=10)
    if got.returncode == 0:
        container = json.loads(got.stdout)[0]
        require(container["Id"] == cid and container["Config"]["Labels"].get(LABEL) == owner,
                "refusing cleanup of a differently owned container")
        raw = cleanup_container(cid)
        result = {"safe": raw["safe"], "container_id": cid,
                  "commands": [{key: command[key] for key in ("argv", "returncode", "error") if key in command}
                               for command in raw.get("commands", [])]}
    else:
        require(any(f"no such {kind}: {cid}" in (got.stdout + got.stderr).lower()
                    for kind in ("object", "container")), "container inspection failed")
        result = {"safe": True, "already_absent": True, "container_id": cid}
    require(result["safe"], "owned container cleanup incomplete")
    require_container_name_available(name)
    return result


def classify_strict(observed, strict, rc):
    require(observed.get("verdict") == "passed" and observed.get("require_attested") is False,
            "actual raw route validation failed")
    require(len(observed["expected_owners"]) == 74 and observed["expected_projection_units"] == 2214,
            "observational census population incomplete")
    if rc == 0:
        require(strict.get("verdict") == "passed" and strict.get("require_attested") is True,
                "strict gate did not attest")
        return "attested"
    require(rc == 1 and strict.get("verdict") == "REFUSED" and len(strict.get("problems", [])) == 1
            and strict["problems"][0].startswith("current contract does not attest every planned dense owner"),
            "unexpected strict gate failure")
    structures = observed["cell_launch_agreement"]["structures"]
    require(set(structures) == {"dense", "routed_moe"}, "unexpected attestation structures")
    for structure, count in (("dense", 52), ("routed_moe", 22)):
        phases = structures[structure]["phases"]
        require(len(phases) == 2 and all(
            row["covered_by_cell"] == (0 if structure == "dense" else count)
            and row["unattested"] == (count if structure == "dense" else 0)
            for row in phases.values()), "strict refusal hides another coverage defect")
    return "dense_cells_unattested"


def container_flags(out, name, owner, cpus):
    return ["--name", name, "--cidfile", str(out / (name + ".cid")), "--label", f"{LABEL}={owner}",
            "--memory=64g", "--memory-swap=64g", "--cpuset-cpus=" + ",".join(map(str, cpus)),
            "--ipc=host", *[item for k, v in LIMITS.items() for item in ("-e", f"{k}={v}")],
            "-v", "/mnt/shared:/mnt/shared:ro"]


def publish_output(out, archive):
    """Archive this completed owned output, excluding existing compilation caches."""
    require(not archive.exists(), "fresh shared result archive required")
    shutil.copytree(out, archive, ignore=shutil.ignore_patterns("ext", "vllm-cache"))
    files = {}
    for path in archive.rglob("*"):
        require(not path.is_symlink(), "unsafe result archive symlink")
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            rel = str(path.relative_to(archive))
            digest = sha(path)
            require(digest == sha(out / rel), "result archive byte drift")
            path.chmod(0o644)
            files[rel] = {"sha256": digest, "bytes": path.stat().st_size}
    archive.chmod(0o755)
    receipt = {"schema": "prismaquant.lfm-mixed-serving-archive.v1", "source": str(out),
               "archive": str(archive), "worker_host": socket.gethostname(), "files": files}
    write(archive / "archive-receipt.json", receipt)
    (archive / "archive-receipt.json").chmod(0o644)
    write(out / "archive-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("out", "encoder", "encoder-manifest", "artifact", "source", "plan", "calibration", "assembly-result"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--archive", type=Path, help="fresh shared result destination after cleanup")
    parser.add_argument("--encoder-manifest-sha256", required=True)
    parser.add_argument("--assembly-result-sha256", required=True)
    parser.add_argument("--seconds", type=int, default=5400)
    parser.add_argument("--port", type=int, default=8198)
    parser.add_argument("--preflight-only", action="store_true",
                        help="verify the complete CPU handoff and seal, then stop before Docker")
    args = parser.parse_args()
    admitted = admission()
    require(60 <= args.seconds <= 7200 and 1024 <= args.port < 65536, "invalid bounds")
    cpus = sorted(os.sched_getaffinity(0))
    require(1 <= len(cpus) <= 4, "reserve at most four CPUs and preserve PB affinity")
    for key in ("out", "encoder", "artifact", "source", "plan", "calibration", "assembly_result", "encoder_manifest"):
        setattr(args, key, getattr(args, key).resolve())
    if args.archive is not None:
        args.archive = args.archive.resolve()
        args.archive.relative_to(Path("/mnt/shared"))
        require(not args.archive.exists(), "fresh shared result archive required")
    for path in (args.artifact, args.source):
        path.relative_to(Path("/mnt/shared"))
    require(not args.out.exists(), "fresh output required")
    require(not any(args.out.is_relative_to(p) for p in (args.artifact, args.source, args.encoder, args.calibration)),
            "output/input collision")
    args.out.mkdir(parents=True)
    for name in ("census", "smoke", "ext", "vllm-cache"):
        (args.out / name).mkdir()
    ts, out = args.encoder, args.out
    bound_encoder = lambda: verify_encoder(ts, args.encoder_manifest, args.encoder_manifest_sha256)
    bound_encoder()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ts / "src"), str(ts)]
    from tessera.serving_parts import source_identity
    from tessera.serving.build_identity import is_complete, incomplete_reason
    from tessera.serving.contract import derive_smoke_status
    # Load the actual mixed gate without constructing a model or importing CUDA.
    spec = importlib.util.spec_from_file_location("mixed_census_gate", ts / "experiments/ts5_census_check.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assembly_record = verify_assembly(args.artifact, args.assembly_result, args.assembly_result_sha256)
    require(sha(args.calibration / "preparation-seal.json") == CALIBRATION_SEAL, "calibration seal changed")
    calibration = read(args.calibration / "preparation-seal.json")
    for name, record in calibration["files"].items():
        require(sha(args.calibration / name) == record["sha256"], "calibration bytes changed")
    plan = read(args.plan)
    require(plan == read(args.calibration / "plan.json"), "calibration common plan changed")
    manifest = read(args.artifact / "tessera_serving_manifest.json")
    teacher = source_identity(args.source)
    identity = validate_handoff(plan, manifest, teacher, calibration)
    config = read(args.artifact / "config.json")
    # tessera 5ac9b72 (#377): the gate's population is the runtime's attested
    # construction entry, resolved here exactly as check_census resolves it.
    entry = gate.construction_entry(config.get("architectures"), gate.load_serving_contract())
    require(entry is not None, "no attested construction entry for the checkpoint architecture")
    gate._population(plan, config, manifest, entry)
    require(sha(ts / "experiments/moe_greedy_smoke_prompts.json") == PROMPTS, "fixed smoke prompts changed")
    checkpoint = source_identity(args.artifact)
    seal = {"checkpoint": str(args.artifact), "checkpoint_identity": checkpoint, "export_identity": identity,
            "encoder": bound_encoder(),
            # The artifact was encoded at manifest["git"]; the serving checkout
            # is ENCODER. They may differ when only experiments/ moved: the
            # runtime compares the encoder fixture id at load
            # (tessera.serving.sharding), so the stamp records both commits.
            "artifact_encoder_git": manifest.get("git"),
            "assembly_stdout_sha256": args.assembly_result_sha256,
            "assembly_record_sha256": hashlib.sha256(json.dumps(assembly_record, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "calibration_seal_sha256": CALIBRATION_SEAL, "plan_sha256": sha(args.plan)}
    write(out / "artifact-seal.json", seal)
    if args.preflight_only:
        receipt = {"schema": "prismaquant.lfm-mixed-serving-preflight.v1", "status": "verified",
                   "artifact_seal_sha256": sha(out / "artifact-seal.json"),
                   "expected_owners": 74, "expected_projection_units": 2214,
                   "gpu_or_container_launched": False, "out": str(out)}
        write(out / "preflight.json", receipt)
        if args.archive is not None:
            publish_output(out, args.archive)
        print(json.dumps(receipt))
        return
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE]))[0]
    require(IMAGE in image.get("RepoDigests", []), "daemon lacks exact EUGR image")
    owner = uuid.uuid4().hex
    prefix = "lfm-mixed-" + owner[:12]
    names = [prefix + "-" + phase for phase in ("census", "bf16", "tessera")]
    for name in names:
        require_container_name_available(name)
    state = {"schema": "prismaquant.lfm-mixed-serving.v1", "status": "running", "owner": owner,
             "admission": admitted,
             "source_snapshot": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
             "image": {"Id": image["Id"], "RepoDigests": image["RepoDigests"],
                       **{key + "_sha256": hashlib.sha256(json.dumps(image[key], sort_keys=True,
                          separators=(",", ":")).encode()).hexdigest() for key in ("Config", "RootFS")}}, "cpu_affinity": cpus, "encoder": seal["encoder"], "phases": {}, "cleanup": {},
             "started_unix": time.time(), "worker_host": socket.gethostname(), "telemetry_success": {}, "telemetry_errors": []}
    deadline = time.monotonic() + args.seconds
    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("mixed serving deadline exceeded")
        return value
    def interrupt(_sig, _frame):
        raise TimeoutError("mixed serving interrupted")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    stopped = threading.Event()
    def telemetry():
        with (out / "telemetry.jsonl").open("x", buffering=1) as stream:
            while not stopped.is_set():
                row = {"unix": time.time(), "meminfo": Path("/proc/meminfo").read_text(),
                       "cpu_stat": Path("/proc/stat").read_text()}
                try:
                    row["power"] = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], text=True, timeout=3)
                    for host in ("sparky", "sparklina"):
                        row[host] = _http(f"http://{host}:19999/api/v1/allmetrics?format=json", timeout=3)
                        require(bool(row[host]), "empty Netdata")
                        state["telemetry_success"][host] = True
                except Exception as exc:
                    state["telemetry_errors"].append(repr(exc))
                stream.write(json.dumps(row) + "\n")
                stopped.wait(5)
    monitor = threading.Thread(target=telemetry, daemon=True)
    env = {**os.environ, **LIMITS, "TS": str(ts), "RUNS": str(out), "EXT": str(out / "ext"), "IMG": IMAGE,
           "RUNTIME_IMAGE_PY": sys.executable, "BUILD_IDENTITY_PY": sys.executable,
           "PYTHONPATH": str(ts / "src"), "PYTHONDONTWRITEBYTECODE": "1", "TESSERA_COMMIT": ENCODER}
    def frozen():
        bound_encoder()
        require(source_identity(args.artifact) == checkpoint and source_identity(args.source) == teacher,
                "served artifact/source identity drift")
        require(sha(args.plan) == seal["plan_sha256"], "common plan drift")
    def run(label, command, allowed=(0,)):
        state["phases"][label] = {"argv": command, "started": time.time()}
        with (out / (label + ".log")).open("x") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining())
        state["phases"][label].update(returncode=result.returncode, finished=time.time())
        require(result.returncode in allowed, f"{label} failed: {result.returncode}")
        return result.returncode
    def cleanup(name):
        result = cleanup_owned(name, out / (name + ".cid"), owner)
        state["cleanup"].setdefault(name, []).append(result)
    def container(label, command, body=None):
        name = prefix + "-" + label
        require_container_name_available(name)
        frozen()
        state["phases"][label] = {"argv": command, "started": time.time()}
        proc = None
        try:
            with (out / ("serve_" + label + ".log")).open("x") as log:
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                cidpath = out / (name + ".cid")
                while not cidpath.exists() or not re.fullmatch(r"[0-9a-f]{64}", cidpath.read_text().strip()):
                    require(proc.poll() is None, f"{label} stopped before container creation")
                    time.sleep(min(0.2, remaining()))
                cid = cidpath.read_text().strip()
                inspected = json.loads(subprocess.check_output(["docker", "inspect", cid], timeout=10))[0]
                state["phases"][label]["container"] = public_container(inspected)
                verify_owned(inspected, cid, owner, cpus)
                if body is not None:
                    ready = False
                    while not ready:
                        require(proc.poll() is None, f"{label} service exited")
                        try:
                            ready = bool(_http(f"http://127.0.0.1:{args.port}/v1/models", timeout=2))
                        except Exception:
                            pass
                        if not ready:
                            time.sleep(min(2, remaining()))
                    body()
                else:
                    require(proc.wait(timeout=remaining()) == 0, f"{label} failed")
        finally:
            cleanup(name)
            if proc is not None:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=10)
            state["phases"][label]["finished"] = time.time()
            state["phases"][label]["launcher_returncode"] = None if proc is None else proc.returncode
        frozen()
    monitor.start()
    try:
        launcher = str(ts / "experiments/tessera_plugin_run.sh")
        census_cmd = shlex.join(["python3", "tools/tessera_route_census.py", str(args.artifact), "/census/census.json",
                                "--expect-modules", "74", "--prompt-tokens", "64", "--max-model-len", "512",
                                "--gpu-memory-utilization", "0.35", "--tessera-commit", ENCODER])
        census_cmd += ' --runtime-image "$TESSERA_CENSUS_RUNTIME_IMAGE"'
        container("census", [launcher, *container_flags(out, names[0], owner, cpus),
                  "-e", "TESSERA_SERVE_MODE=resident", "-v", f"{out / 'census'}:/census", "--", census_cmd])
        base_gate = [sys.executable, str(ts / "experiments/ts5_census_check.py"), "--plan", str(args.plan),
                     "--checkpoint", str(args.artifact), "--census", str(out / "census/census.json"), "--runtime-image", IMAGE]
        run("observed-check", [*base_gate, "--out", str(out / "census/observed-check.json")])
        rc = run("strict-check", [*base_gate, "--require-attested", "--out", str(out / "census/strict-check.json")], (0, 1))
        state["attestation"] = classify_strict(read(out / "census/observed-check.json"), read(out / "census/strict-check.json"), rc)
        smoke = ts / "experiments/moe_greedy_smoke.py"
        run("smoke-preflight", [sys.executable, str(smoke), "preflight"])
        for arm, model, name in zip(("bf16", "tessera"), (args.source, args.artifact), names[1:]):
            write(out / f"smoke/identity_{arm}_before.json", source_identity(model))
            server = ["vllm", "serve", str(model), "--served-model-name", "kl-target", "--host", "0.0.0.0", "--port", "8000",
                      "--max-model-len", "4096", "--max-num-seqs", "8", "--gpu-memory-utilization", "0.35",
                      "--max-logprobs", "1024", "--enforce-eager", "--trust-remote-code", "--tensor-parallel-size", "1"]
            flags = container_flags(out, name, owner, cpus) + ["-p", f"127.0.0.1:{args.port}:8000", "-v", f"{out / 'vllm-cache'}:/root/.cache/vllm"]
            command = (["docker", "run", "--rm", "--pull=never", "--gpus", "all", *flags, "--entrypoint", "vllm", IMAGE, *server[1:]]
                       if arm == "bf16" else [launcher, *flags, "-e", "TESSERA_SERVE_MODE=resident", "--", shlex.join(server)])
            def requests():
                run(f"metrics-{arm}", ["bash", "-c", 'source "$1"; serve_require_no_spec_decode "$2" "$3"', "bash",
                    str(ts / "experiments/serve_metrics.sh"), str(args.port), str(out / f"smoke/metrics_{arm}.txt")])
                run(f"smoke-{arm}", [sys.executable, str(smoke), "run", "--url", f"http://127.0.0.1:{args.port}/v1/completions",
                    "--tokenizer", str(args.artifact), "--prompts", str(ts / "experiments/moe_greedy_smoke_prompts.json"),
                    "--arm", arm, "--out", str(out / f"smoke/smoke_{arm}.json"), "--pure-greedy"])
            with socket.socket() as port_check:
                port_check.bind(("127.0.0.1", args.port))
            container(arm, command, requests)
            write(out / f"smoke/identity_{arm}_after.json", source_identity(model))
            run(f"build-{arm}", ["bash", "-c", 'source "$1/experiments/runtime_image.sh"; source "$1/experiments/build_identity.sh"; runtime_image_require "$2"; build_identity_stamp "$3" "$4" "$5" "$2" "$6" 1 "$7"', "bash",
                str(ts), IMAGE, str(out / f"serve_{arm}.log"), str(out / f"smoke/smoke_{arm}.build.json"),
                str(out / "vllm-cache"), "resident" if arm == "tessera" else "", str(model)])
            build = read(out / f"smoke/smoke_{arm}.build.json")
            require(is_complete(build), incomplete_reason(build))
            require(build["identity"]["image"] == IMAGE and build["identity"]["eager"] is True
                    and build["identity"]["compiled_forward"] is False, "serving build/runtime mismatch")
            if arm == "tessera":
                require(build["identity"]["serve_mode"] == "resident", "student residency drift")
        run("compare", [sys.executable, str(smoke), "compare", str(out / "smoke/smoke_bf16.json"), str(out / "smoke/smoke_tessera.json"),
            "--out", str(out / "smoke/pair.json"), "--markdown", str(out / "smoke/pair.md"), "--subject", "tessera", "--reference", "bf16_source"])
        state["smoke_status"] = derive_smoke_status(read(out / "smoke/pair.json")["contract_record"])
        frozen()
        state["status"] = "observed"
        state["sanity_passed"] = state["smoke_status"] == "recorded"
        state["production_admitted"] = False
        state["quality_or_speed_claimed"] = False
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        stopped.set()
        monitor.join(timeout=15)
        failures = []
        for name in names:
            try:
                cleanup(name)
            except Exception as exc:
                failures.append(repr(exc))
        state["cleanup_errors"] = failures
        state["finished_unix"] = time.time()
        write(out / "host-status.json", state)
        if args.archive is not None and not failures:
            publish_output(out, args.archive)
    require(not state["cleanup_errors"] and set(state["telemetry_success"]) == {"sparky", "sparklina"},
            "cleanup or both-host telemetry incomplete")
    require(state["sanity_passed"], "paired smoke did not record a passing bounded sanity observation")
    print(json.dumps({"status": state["status"], "attestation": state["attestation"], "smoke_status": state["smoke_status"], "out": str(out)}))


if __name__ == "__main__":
    main()
