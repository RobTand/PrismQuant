"""PB measurement of the shared activation adapter on retained native inputs."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    from safetensors.torch import load_file
    from prismaquant import fp8_dynamic
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensors", required=True, type=Path)
    parser.add_argument("--tensors-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seconds", default=20., type=float)
    args = parser.parse_args()
    if digest(args.tensors) != args.tensors_sha256:
        raise ValueError("native inputs changed")
    args.out.mkdir(parents=True, exist_ok=False)
    tensors = load_file(str(args.tensors), device="cuda")
    report = {"schema": "prismaquant.fp8_activation_profile.v1", "input_sha256": args.tensors_sha256,
        "source_sha256": digest(inspect.getfile(fp8_dynamic)), "host": socket.gethostname(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
        "affinity": sorted(os.sched_getaffinity(0)), "scope": "shared FP8 activation QDQ only; no model throughput claim",
        "phases": {}}
    for phase in ("prefill", "decode"):
        x = tensors[phase + ".input"]
        def call():
            return fp8_dynamic.fp8_dynamic_activation_qdq_vllm(x).dequant.to(x.dtype)
        for _ in range(20):
            call()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        count = 0
        while time.time() - started < args.seconds:
            for _ in range(32):
                call()
            count += 32
            torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()
        stopped = time.time()
        elapsed = begin.elapsed_time(end)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(32):
                with torch.profiler.record_function("shared_fp8_activation_qdq"):
                    call()
            torch.cuda.synchronize()
        trace = args.out / (phase + ".trace.json")
        prof.export_chrome_trace(str(trace))
        table = args.out / (phase + ".profile.txt")
        table.write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
        kernels = [{"name": row.key, "count": row.count, "self_cpu_us": row.self_cpu_time_total,
                    "self_device_us": row.self_device_time_total} for row in prof.key_averages()]
        report["phases"][phase] = {"shape": list(x.shape), "dtype": str(x.dtype), "calls": count,
            "window_unix": [started, stopped], "wall_seconds": stopped - started,
            "cuda_elapsed_ms": elapsed, "cuda_ms_per_call": elapsed / count,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "trace_sha256": digest(trace),
            "profile_table_sha256": digest(table), "profiled_calls": 32, "operators": kernels}
    target = args.out / "profile.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(target), "sha256": digest(target), "phases": {
        name: {k: v for k, v in value.items() if k != "operators"} for name, value in report["phases"].items()}}))


if __name__ == "__main__":
    main()
