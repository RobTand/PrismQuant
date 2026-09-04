#!/usr/bin/env python3
"""Bounded, opt-in BF16 instrument campaign; execute only as a PrismaBuild action.

Host freezes content before one container starts. The container runs one eager
server and journals every HTTP response, including an interrupted cap-growth
schedule. A completed observation is not a shipping verdict or a speed claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("physical campaign deadline exhausted")
    return value


def require_frozen_content(expected, observed):
    if expected != observed:
        raise ValueError("client source differs from pre-launch frozen artifact")


def server_argv(nonce):
    return ["vllm", "serve", "/model", "--host", "127.0.0.1", "--port", "8187",
            "--served-model-name", nonce, "--dtype", "bfloat16",
            "--kv-cache-memory-bytes", str(1 << 30), "--max-model-len", "4096",
            "--max-num-seqs", "1", "--max-num-batched-tokens", "512",
            "--enforce-eager", "--generation-config", "vllm"]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def cleanup_container(name):
    evidence = {"safe": False, "commands": []}
    for command in (["docker", "stop", "--time", "10", name],
                    ["docker", "rm", "-f", name], ["docker", "inspect", name]):
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=15)
            evidence["commands"].append({"argv": command, "returncode": result.returncode,
                                         "output": result.stdout})
            if command[1] == "inspect":
                if result.returncode == 0:
                    evidence["safe"] = json.loads(result.stdout)[0]["State"]["Running"] is False
                elif "No such object" in result.stdout or "No such container" in result.stdout:
                    evidence["safe"] = True
        except Exception as exc:
            evidence["commands"].append({"argv": command, "error": repr(exc)})
    return evidence


def bootstrap():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    from prismaquant_source_bootstrap import activate_prismaquant_source, _install_exact_package_namespace
    root = activate_prismaquant_source()
    _install_exact_package_namespace(root)
    import measure_boundary_control as client
    from prismaquant import boundary_control as bc
    return root, client, bc


def client_arm(args):
    root, client, bc = bootstrap()
    expected = json.loads(Path("/run/prelaunch.json").read_text())["content"]
    capture = client._capture_artifact
    def bound_capture(model_dir):
        content, stats = capture(model_dir)
        require_frozen_content(expected, content)
        return content, stats
    client._capture_artifact = bound_capture
    # This process already verified and installed this exact source namespace.
    client.activate_prismaquant_source = lambda: root
    client._install_exact_package_namespace = lambda _root: None
    post = bc.vqm._post_json
    step = bc.measure_step
    with open(f"/run/{args.role}-http.jsonl", "x", buffering=1) as journal:
        def recorded(url, body):
            began = time.time()
            try:
                response = post(url, body)
            except BaseException as exc:
                journal.write(json.dumps({"url": url, "request": body, "started": began,
                                          "finished": time.time(), "error": repr(exc)}) + "\n")
                raise
            journal.write(json.dumps({"url": url, "request": body, "response": response,
                                      "started": began, "finished": time.time()},
                                     ensure_ascii=False, allow_nan=False) + "\n")
            return response
        bc.vqm._post_json = recorded
        with open(f"/run/{args.role}-steps.jsonl", "x", buffering=1) as steps:
            def recorded_step(*pos, **kwargs):
                value = step(*pos, **kwargs)
                steps.write(json.dumps({"raw": bool(kwargs.get("raw")), "step": value},
                                       ensure_ascii=False, allow_nan=False) + "\n")
                return value
            bc.measure_step = recorded_step
            argv = [args.role, "--base-url", "http://127.0.0.1:8187",
                    "--model-name", args.nonce, "--model-dir", "/model", "--image", args.image,
                    "--campaign-id", args.nonce, "--out", f"/run/{args.role}.json"]
            argv += (["--contract", "/repo/experiments/pq87_boundary_contract.json", "--legacy-ab"]
                     if args.role == "control" else ["--control", "/run/control.json"])
            return client.main(argv)


def _http(url, body=None, timeout=2):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def inside(args):
    deadline = time.monotonic() + args.seconds
    status = {"status": "inconclusive", "started": time.time(), "server_argv": server_argv(args.nonce)}
    server = None
    stop = threading.Event()
    def interrupted(_signal, _frame):
        raise TimeoutError("physical campaign terminated")
    signal.signal(signal.SIGTERM, interrupted)
    def process_io():
        sys.path.insert(0, "/repo/tools")
        import serve_fingerprint as sf
        with open("/run/process-io.jsonl", "x", buffering=1) as log:
            while not stop.is_set():
                row = {"time": time.time(), "processes": []}
                for pid in sf.find_server_pids():
                    try:
                        row["processes"].append({"pid": pid,
                            "io": Path(f"/proc/{pid}/io").read_text(),
                            "stat": Path(f"/proc/{pid}/stat").read_text(),
                            "status": Path(f"/proc/{pid}/status").read_text()})
                    except OSError as exc:
                        row["processes"].append({"pid": pid, "error": repr(exc)})
                log.write(json.dumps(row) + "\n")
                stop.wait(10)
    monitor = threading.Thread(target=process_io, daemon=True)
    try:
        with open("/run/server.log", "x") as server_log:
            server = subprocess.Popen(server_argv(args.nonce), stdout=server_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            monitor.start()
            while True:
                remaining(deadline)
                if server.poll() is not None:
                    raise RuntimeError(f"server exited {server.returncode}")
                try:
                    _http("http://127.0.0.1:8187/health")
                    break
                except Exception:
                    time.sleep(min(1, remaining(deadline)))
            # Warm both request paths before any process/residency fingerprint.
            for raw in (True, False):
                body = {"model": args.nonce, "max_tokens": 1, "temperature": 0,
                        **({"prompt": "Warm up."} if raw else {
                            "messages": [{"role": "user", "content": "Warm up."}]})}
                endpoint = "completions" if raw else "chat/completions"
                dump(Path(f"/run/warm-{'raw' if raw else 'chat'}.json"),
                     _http("http://127.0.0.1:8187/v1/" + endpoint, body,
                           timeout=min(60, remaining(deadline))))
            for role in ("control", "candidate"):
                status["active_role"] = role
                dump(Path("/run/inside-status.json"), status)
                with open(f"/run/{role}-client.log", "x") as client_log:
                    result = subprocess.run([sys.executable, "-P", __file__, "--client",
                        "--role", role, "--image", args.image, "--nonce", args.nonce],
                        stdout=client_log, stderr=subprocess.STDOUT, timeout=remaining(deadline))
                status[role + "_returncode"] = result.returncode
                if result.returncode != 0:
                    raise RuntimeError(f"{role} incomplete or refused: exit {result.returncode}")
            status["status"] = "completed"
    except BaseException as exc:
        status["error"] = repr(exc)
    finally:
        stop.set()
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=5)
        status["finished"] = time.time()
        dump(Path("/run/inside-status.json"), status)
    return 0  # Completed observation, not a success verdict; PB must not retry a censored arm.


def host(args):
    root, client, bc = bootstrap()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    def interrupted(_signal, _frame):
        raise TimeoutError("physical campaign terminated")
    signal.signal(signal.SIGTERM, interrupted)
    deadline = time.monotonic() + min(args.seconds, 1200)
    status = {"status": "inconclusive", "started": time.time(), "image": args.image,
              "source_snapshot": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                         text=True).strip(),
              "deadline_seconds": min(args.seconds, 1200), "no_speed_claim": True}
    nonce = out.name
    container = "pq87-" + nonce
    stop = threading.Event()
    def telemetry():
        with (out / "telemetry.jsonl").open("x", buffering=1) as log:
            while not stop.is_set():
                row = {"time": time.time(), "meminfo": Path("/proc/meminfo").read_text()}
                try:
                    row["nvidia_smi"] = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=power.draw,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], text=True, timeout=3).strip()
                except Exception as exc:
                    row["nvidia_smi_error"] = repr(exc)
                for node in ("127.0.0.1", "192.168.1.110"):
                    try:
                        row[node] = _http(f"http://{node}:19999/api/v1/allmetrics?format=json", timeout=2)
                    except Exception as exc:
                        row[node] = {"error": repr(exc)}
                log.write(json.dumps(row) + "\n")
                stop.wait(10)
    monitor = threading.Thread(target=telemetry, daemon=True)
    try:
        inspect = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image],
                                                     text=True, timeout=min(10, remaining(deadline))))[0]
        if args.image not in inspect.get("RepoDigests", []):
            raise ValueError("Docker image has no matching immutable RepoDigest")
        dump(out / "image.json", inspect)
        before, source_stats = client._capture_artifact(args.model_dir)
        frozen = out / "frozen-model"
        shutil.copytree(args.model_dir, frozen)
        content, _ = client._capture_artifact(frozen)
        after, after_stats = client._capture_artifact(args.model_dir)
        require_frozen_content(before, content)
        if before != after or source_stats != after_stats:
            raise ValueError("source changed while creating frozen artifact")
        dump(out / "prelaunch.json", {"content": content, "artifact_id": bc.artifact_content_id(content),
                                     "source_stats": source_stats, "captured": time.time()})
        # Read-only mount is the loaded-byte boundary; these modes prevent ordinary host writes too.
        for path in frozen.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        frozen.chmod(0o555)
        for directory in ("tmp", "triton", "inductor", "vllm-cache"):
            (out / directory).mkdir()
        monitor.start()
        command = ["docker", "run", "--rm", "--pull=never", "--name", container,
            "--gpus", "all", "--memory=28g", "--memory-swap=28g", "--cpus=2",
            "--shm-size=1g", "--network=host", "--cap-add=SYS_PTRACE",
            "--security-opt=seccomp=unconfined", "-v", f"{root}:/repo:ro",
            "-v", f"{frozen}:/model:ro", "-v", f"{out}:/run",
            "-e", "SPT_NOENV=1", "-e", "OMP_NUM_THREADS=2",
            "-e", "TMPDIR=/run/tmp", "-e", "TRITON_CACHE_DIR=/run/triton",
            "-e", "TORCHINDUCTOR_CACHE_DIR=/run/inductor", "-e", "VLLM_CACHE_ROOT=/run/vllm-cache",
            "-e", "HF_HUB_OFFLINE=1", "-e", "PYTHONSAFEPATH=1",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONNOUSERSITE=1",
            "-e", "PQ_RUNTIME_PRISMAQUANT_ROOT=/repo", "--entrypoint", "/usr/bin/env",
            args.image, "-u", "PYTHONPATH", "python3", "-P",
            "/repo/experiments/pq87_physical_ab.py", "--inside",
            "--seconds", str(max(1, int(remaining(deadline)) - 15)), "--image", args.image,
            "--nonce", nonce]
        status["docker_argv"] = command
        dump(out / "campaign.json", status)
        with (out / "container.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           timeout=remaining(deadline), check=True)
        status["inside"] = json.loads((out / "inside-status.json").read_text())
        status["status"] = status["inside"]["status"]
    except BaseException as exc:
        status["error"] = repr(exc)
    finally:
        stop.set()
        # The exact unique container is ours; never stop or inspect another campaign's process.
        status["cleanup"] = cleanup_container(container)
        if not status["cleanup"]["safe"]:
            status["status"] = "inconclusive"
            status["cleanup_error"] = "exact container not confirmed absent or stopped"
        status["finished"] = time.time()
        dump(out / "campaign.json", status)
        print(json.dumps({"out": str(out), **status}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-dir", default="/home/rob/models/Qwen3-0.6B")
    parser.add_argument("--out")
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--nonce")
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--role", choices=("control", "candidate"))
    args = parser.parse_args()
    if args.client:
        return client_arm(args)
    if args.inside:
        return inside(args)
    if not args.out:
        parser.error("host invocation requires --out")
    return host(args)


if __name__ == "__main__":
    raise SystemExit(main())
