#!/usr/bin/env python3
"""Frozen sequential native BF16/quantized validation, only through PrismaBuild.

Reuses the instrument's server configuration, content guard, cleanup and
measurement client. Screen and disjoint heldout schedules are inputs, never
selected after reading a candidate. A completed run is not a ship decision.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pq87_physical_ab as instrument


def validate_manifest(value):
    from prismaquant import boundary_control as bc
    if not isinstance(value, dict) or set(value) != {
        "schema", "image", "models", "contracts", "deadline_seconds", "netdata_urls"
    } or value["schema"] != "prismaquant.boundary_validation/1":
        raise ValueError("validation manifest fields differ from schema")
    if not isinstance(value["image"], str) or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", value["image"]):
        raise ValueError("validation image must be an immutable digest")
    models = value["models"]
    if (not isinstance(models, dict) or set(models) != {"control", "candidate"}
            or any(not isinstance(path, str) or not Path(path).is_absolute() for path in models.values())
            or models["control"] == models["candidate"]):
        raise ValueError("validation requires distinct explicit control/candidate model paths")
    contracts = value["contracts"]
    if not isinstance(contracts, dict) or set(contracts) != {"screen", "heldout"}:
        raise ValueError("validation requires screen and heldout contracts")
    for contract in contracts.values():
        bc.validate_contract(contract)
    if {p["text"] for p in contracts["screen"]["prompts"]} & {
            p["text"] for p in contracts["heldout"]["prompts"]}:
        raise ValueError("screen and heldout prompts overlap")
    if type(value["deadline_seconds"]) is not int or value["deadline_seconds"] <= 0:
        raise ValueError("deadline_seconds must be a positive integer")
    urls = value["netdata_urls"]
    if not isinstance(urls, list) or len(urls) != 2 or len(set(urls)) != 2:
        raise ValueError("declare both distinct box telemetry URLs")
    for url in urls:
        parsed = urlsplit(url)
        if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError("invalid Netdata URL")
    return value


def server_argv(nonce, model):
    argv = instrument.server_argv(nonce)
    argv[argv.index("serve") + 1] = model
    return argv


def completed_candidate(receipt, returncode):
    verdict = receipt.get("decision", {}).get("verdict")
    return (verdict == "accepted" and returncode == 0) or (verdict == "refused" and returncode == 2)


def require_ppl_binding(before, model, nonce):
    if not before.get("residency_readable") or not before.get("serve_session_id"):
        raise ValueError("PPL requires readable live residency and serve session")
    if Path(before["model"]).resolve() != Path(model).resolve() or before["models_endpoint_binding"]["model"]["id"] != nonce:
        raise ValueError("PPL source differs from observed server")


def require_ppl_measurement(result, prompt_count):
    """A measured threshold refusal is data; missing/skipped logprobs are not."""
    metrics = result.metrics
    if (result.name != "perplexity" or type(result.passed) is not bool
            or not isinstance(metrics, dict) or metrics.get("skipped", False) is not False
            or metrics.get("spec_decode_detected") is not False):
        raise ValueError("PPL measurement is missing, skipped, or speculative")
    per_prompt = metrics.get("per_prompt_avg_nll")
    tokens = metrics.get("n_tokens")
    if (not isinstance(per_prompt, list) or len(per_prompt) != prompt_count
            or type(tokens) is not int or tokens < prompt_count):
        raise ValueError("PPL measurement does not cover the full prompt roster with scored tokens")
    values = [metrics.get(key) for key in ("perplexity", "mean_nll_per_tok",
                                          "p99_nll_per_tok", "max_nll_per_tok")]
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in values + per_prompt) or values[0] <= 0):
        raise ValueError("PPL measurement has missing or nonfinite numeric metrics")
    return result


def require_container_name_available(name):
    result = subprocess.run(["docker", "inspect", name], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=10)
    if result.returncode == 0:
        raise ValueError("validation container name already exists")
    if not any(f"no such {kind}: {name}" in result.stdout.lower() for kind in ("object", "container")):
        raise ValueError("cannot establish validation container name is unused")


def cleanup_owned_container(cidfile, name, owner, nonce, *, attempted):
    """Never stop by a potentially borrowed name, even after failed creation."""
    evidence = {"safe": not attempted, "commands": [], "attempted": attempted}
    if not attempted:
        return evidence
    try:
        cid = cidfile.read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{64}", cid):
            raise ValueError("Docker did not publish an exact container ID")
        inspected = subprocess.run(["docker", "inspect", cid], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, timeout=10)
        evidence["commands"].append({"argv": ["docker", "inspect", cid],
            "returncode": inspected.returncode, "output": inspected.stdout})
        if inspected.returncode != 0:
            if any(f"no such {kind}: {cid}" in inspected.stdout.lower() for kind in ("object", "container")):
                evidence["safe"] = True
                return evidence
            raise ValueError("cannot inspect created container")
        observed = json.loads(inspected.stdout)[0]
        labels = observed.get("Config", {}).get("Labels") or {}
        if (observed.get("Id") != cid or observed.get("Name") != "/" + name
                or labels.get("prismabuild.action") != owner
                or labels.get("prismaquant.pq87.campaign") != nonce):
            raise ValueError("created container ownership differs")
        cleanup = instrument.cleanup_container(cid)
        evidence["commands"].extend(cleanup["commands"])
        evidence.update(safe=cleanup["safe"], container_id=cid)
    except Exception as exc:
        evidence["error"] = repr(exc)
    return evidence


def _interrupt(_signal, _frame):
    raise TimeoutError("paired validation terminated")


def client_phase(args):
    root, client, bc = instrument.bootstrap()
    import serve_fingerprint as sf
    manifest = json.loads(Path("/run/manifest.json").read_text())
    model = f"/models/{args.role}"
    expected = json.loads(Path(f"/run/prelaunch-{args.role}.json").read_text())["content"]
    capture = client._capture_artifact
    def bound_capture(model_dir):
        content, stats = capture(model_dir)
        instrument.require_frozen_content(expected, content)
        return content, stats
    client._capture_artifact = bound_capture
    client.activate_prismaquant_source = lambda: root
    client._install_exact_package_namespace = lambda _root: None
    prefix = f"/run/{args.role}-{args.population}"
    post, step = bc.vqm._post_json, bc.measure_step
    with open(prefix + "-http.jsonl", "x", buffering=1) as journal, open(
            prefix + "-steps.jsonl", "x", buffering=1) as steps:
        def recorded(url, body):
            row = {"url": url, "request": body, "started": time.time()}
            try:
                response = post(url, body)
                row["response"] = response
                return response
            except BaseException as exc:
                row["error"] = repr(exc)
                raise
            finally:
                row["finished"] = time.time()
                journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        def recorded_step(*pos, **kwargs):
            result = step(*pos, **kwargs)
            steps.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            return result
        bc.vqm._post_json, bc.measure_step = recorded, recorded_step
        if args.population == "ppl":
            before = sf.collect_manifest(image=manifest["image"], base_url="http://127.0.0.1:8187", attestation_phase="pre")
            content, stats = bound_capture(model)
            require_ppl_binding(before, model, args.nonce)
            result = bc.vqm.check_perplexity("http://127.0.0.1:8187", args.nonce,
                bc.vqm.DEFAULT_MAX_PPL, bc.vqm.DEFAULT_MAX_P99_NLL, bc.vqm.DEFAULT_MAX_MEAN_NLL)
            # Preserve even incomplete results as evidence, but never call an
            # empty/skipped CheckResult a completed numeric observation.
            instrument.dump(Path(prefix + "-result.json"), asdict(result))
            require_ppl_measurement(result, len(bc.vqm.EVAL_PROMPTS))
            after = sf.collect_manifest(image=manifest["image"], base_url="http://127.0.0.1:8187", attestation_phase="post")
            require_ppl_binding(after, model, args.nonce)
            after_content, after_stats = bound_capture(model)
            if content != after_content or stats != after_stats or any(before[key] != after[key]
                    for key in ("serve_session_id", "performance_stack_fingerprint", "models_endpoint_binding")):
                raise ValueError("PPL measurement source or server changed")
            instrument.dump(Path(prefix + ".json"), {"schema": "prismaquant.boundary_ppl_screen/1",
                "advisory_only": True, "artifact_id": bc.artifact_content_id(content),
                "serve_pre": before, "serve_post": after, "result": asdict(result)})
            return 0
        argv = [args.role, "--base-url", "http://127.0.0.1:8187", "--model-name", args.nonce,
                "--model-dir", model, "--image", manifest["image"], "--campaign-id", args.nonce,
                "--out", prefix + ".json"]
        argv += (["--contract", f"/run/{args.population}-contract.json"] if args.role == "control" else
                 ["--control", f"/run/control-{args.population}.json", "--decision-policy", "no-new-failures"])
        return client.main(argv)


def inside(args):
    _root, _client, _bc = instrument.bootstrap()
    import serve_fingerprint as sf
    manifest = validate_manifest(json.loads(Path("/run/manifest.json").read_text()))
    deadline = time.monotonic() + args.seconds
    signal.signal(signal.SIGTERM, _interrupt)
    status = {"status": "inconclusive", "advisory_only": True, "started": time.time(), "phases": {}}
    stop = threading.Event()
    def profile():
        with open("/run/process-io.jsonl", "x", buffering=1) as log:
            while not stop.is_set():
                row = {"time": time.time(), "processes": []}
                for pid in sf.find_server_pids():
                    try:
                        row["processes"].append({"pid": pid, "io": Path(f"/proc/{pid}/io").read_text(),
                            "stat": Path(f"/proc/{pid}/stat").read_text(), "status": Path(f"/proc/{pid}/status").read_text()})
                    except OSError as exc:
                        row["processes"].append({"pid": pid, "error": repr(exc)})
                log.write(json.dumps(row) + "\n")
                stop.wait(10)
    monitor = threading.Thread(target=profile, daemon=True)
    monitor.start()
    server = None
    try:
        for role in ("control", "candidate"):
            if sf.find_server_pids():
                raise ValueError("a server remained before the next sequential model arm")
            role_status = status["phases"][role] = {"argv": server_argv(args.nonce, f"/models/{role}")}
            with open(f"/run/server-{role}.log", "x") as log:
                server = subprocess.Popen(role_status["argv"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while True:
                    instrument.remaining(deadline)
                    if server.poll() is not None:
                        raise RuntimeError(f"{role} server exited {server.returncode}")
                    try:
                        instrument._http("http://127.0.0.1:8187/health")
                        break
                    except Exception:
                        time.sleep(min(1, instrument.remaining(deadline)))
                for raw in (True, False):
                    body = {"model": args.nonce, "max_tokens": 1, "temperature": 0,
                            **({"prompt": "Warm up."} if raw else {"messages": [{"role": "user", "content": "Warm up."}]})}
                    endpoint = "completions" if raw else "chat/completions"
                    instrument.dump(Path(f"/run/warm-{role}-{raw}.json"), instrument._http(
                        "http://127.0.0.1:8187/v1/" + endpoint, body, timeout=min(60, instrument.remaining(deadline))))
                for population in ("screen", "heldout", "ppl"):
                    instrument.dump(Path("/run/inside-status.json"), status)
                    with open(f"/run/{role}-{population}-client.log", "x") as client_log:
                        result = subprocess.run([sys.executable, "-P", __file__, "--client", "--role", role,
                            "--population", population, "--nonce", args.nonce], stdout=client_log,
                            stderr=subprocess.STDOUT, timeout=instrument.remaining(deadline))
                    path = Path(f"/run/{role}-{population}.json")
                    receipt = json.loads(path.read_text()) if path.exists() else {}
                    role_status[population] = {"returncode": result.returncode,
                        "decision": receipt.get("decision"), "ppl": receipt.get("result")}
                    if role == "candidate" and population != "ppl":
                        observed = completed_candidate(receipt, result.returncode)
                    else:
                        observed = bool(receipt) and result.returncode == 0
                    if not observed:
                        raise RuntimeError(f"{role}/{population} did not produce a complete comparable observation")
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=min(20, instrument.remaining(deadline)))
                server = None
                # The subsequent arm must not coexist with an orphaned engine.
                if sf.find_server_pids():
                    raise ValueError(f"{role} left a serving process after shutdown")
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
        instrument.dump(Path("/run/inside-status.json"), status)
    return 0  # The receipt carries refusal/inconclusive; the pool must not retry observations.


def host(args):
    if not os.environ.get("PRISMABUILD_CONTAINER_OWNER"):
        raise ValueError("physical validation must be dispatched through PrismaBuild")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise ValueError("physical validation requires an allocated GPU, not a CPU-only pool action")
    root, client, bc = instrument.bootstrap()
    manifest = validate_manifest(json.loads(Path(args.manifest).read_text()))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + manifest["deadline_seconds"]
    signal.signal(signal.SIGTERM, _interrupt)
    owner = os.environ["PRISMABUILD_CONTAINER_OWNER"]
    ownership_nonce = uuid.uuid4().hex
    container = "pq87-" + owner[:16] + "-" + ownership_nonce
    cidfile = out / "container.cid"
    creation_attempted = False
    status = {"status": "inconclusive", "advisory_only": True, "started": time.time(),
        "container_ownership": {"name": container, "action": owner, "nonce": ownership_nonce},
        "manifest_sha256": bc.digest(manifest), "source_snapshot": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    instrument.dump(out / "manifest.json", manifest)
    stop = threading.Event()
    def telemetry():
        with (out / "telemetry.jsonl").open("x", buffering=1) as log:
            while not stop.is_set():
                row = {"time": time.time(), "meminfo": Path("/proc/meminfo").read_text()}
                try:
                    row["gpu_power_w"] = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits"], text=True, timeout=3).strip()
                except Exception as exc:
                    row["gpu_power_error"] = repr(exc)
                for url in manifest["netdata_urls"]:
                    try:
                        row[url] = instrument._http(url.rstrip("/") + "/api/v1/allmetrics?format=json", timeout=2)
                    except Exception as exc:
                        row[url] = {"error": repr(exc)}
                log.write(json.dumps(row) + "\n")
                stop.wait(10)
    try:
        inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", manifest["image"]],
            text=True, timeout=min(10, instrument.remaining(deadline))))[0]
        if manifest["image"] not in inspected.get("RepoDigests", []):
            raise ValueError("Docker image lacks the requested immutable digest")
        instrument.dump(out / "image.json", inspected)
        for role, model in manifest["models"].items():
            instrument.remaining(deadline)
            config = json.loads((Path(model) / "config.json").read_text())
            if role == "control":
                bc.require_bf16_control(config, "bfloat16", None)
            elif (config.get("quantization_config") or {}).get("quant_method") != "compressed-tensors":
                raise ValueError("this first validation manifest requires a native compressed-tensors candidate")
            before, stats = client._capture_artifact(model)
            frozen = out / "models" / role
            shutil.copytree(model, frozen)
            content, _ = client._capture_artifact(frozen)
            after, after_stats = client._capture_artifact(model)
            instrument.require_frozen_content(before, content)
            if before != after or stats != after_stats:
                raise ValueError("source changed while freezing")
            instrument.dump(out / f"prelaunch-{role}.json", {"content": content,
                "artifact_id": bc.artifact_content_id(content), "source_stats": stats, "captured": time.time()})
            for path in frozen.rglob("*"):
                path.chmod(0o555 if path.is_dir() else 0o444)
            frozen.chmod(0o555)
        for population, contract in manifest["contracts"].items():
            instrument.dump(out / f"{population}-contract.json", contract)
        for directory in ("tmp", "triton", "inductor", "vllm-cache"):
            (out / directory).mkdir()
        threading.Thread(target=telemetry, daemon=True).start()
        require_container_name_available(container)
        command = ["docker", "run", "--pull=never", "--name", container,
            "--cidfile", str(cidfile), "--label", "prismaquant.pq87.campaign=" + ownership_nonce,
            "--gpus", "all", "--memory=28g", "--memory-swap=28g", "--cpus=2", "--shm-size=1g",
            "--network=host", "--cap-add=SYS_PTRACE", "--security-opt=seccomp=unconfined",
            "-v", f"{root}:/repo:ro", "-v", f"{out}:/run", "-v", f"{out / 'models'}:/models:ro",
            "-v", f"{out / 'models'}:/run/models:ro",
            "-e", "SPT_NOENV=1", "-e", "OMP_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2",
            "-e", "TMPDIR=/run/tmp", "-e", "TRITON_CACHE_DIR=/run/triton", "-e", "TORCHINDUCTOR_CACHE_DIR=/run/inductor",
            "-e", "VLLM_CACHE_ROOT=/run/vllm-cache", "-e", "HF_HUB_OFFLINE=1", "-e", "PYTHONSAFEPATH=1",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONNOUSERSITE=1", "-e", "PQ_RUNTIME_PRISMAQUANT_ROOT=/repo",
            "--entrypoint", "/usr/bin/env", manifest["image"], "-u", "PYTHONPATH", "python3", "-P",
            "/repo/experiments/pq87_paired_validation.py", "--inside", "--seconds",
            str(max(1, int(instrument.remaining(deadline)) - 30)), "--nonce", out.name]
        status["docker_argv"] = command
        instrument.dump(out / "campaign.json", status)
        with (out / "container.log").open("x") as log:
            creation_attempted = True
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           timeout=instrument.remaining(deadline), check=True)
        status["inside"] = json.loads((out / "inside-status.json").read_text())
        status["status"] = status["inside"]["status"]
    except BaseException as exc:
        status["error"] = repr(exc)
    finally:
        stop.set()
        status["cleanup"] = cleanup_owned_container(cidfile, container, owner, ownership_nonce,
                                                     attempted=creation_attempted)
        if not status["cleanup"]["safe"]:
            status["status"] = "inconclusive"
        status["finished"] = time.time()
        instrument.dump(out / "campaign.json", status)
        print(json.dumps({"out": str(out), **status}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--out")
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--role", choices=("control", "candidate"))
    parser.add_argument("--population", choices=("screen", "heldout", "ppl"))
    parser.add_argument("--nonce")
    parser.add_argument("--seconds", type=int)
    args = parser.parse_args()
    if args.client:
        return client_phase(args)
    if args.inside:
        return inside(args)
    if not args.manifest or not args.out:
        parser.error("host invocation requires --manifest and --out")
    return host(args)


if __name__ == "__main__":
    raise SystemExit(main())
