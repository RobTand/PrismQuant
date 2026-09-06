"""Exercise the actual serving image's dense arithmetic before a mixed export.

This is an ABI and shape check with deterministic synthetic operands. It does
not encode a model, qualify a serving cell, or supply allocator runtime prices.
The final model still owes byte validation, census and paired generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--tessera-commit", required=True)
    args = parser.parse_args()

    import torch
    import vllm
    from safetensors import safe_open
    from tessera.serving import native_ops

    if args.out.exists():
        raise ValueError("preflight output exists; retain it and use a new attempt")
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(253)
    if torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("this declared preflight requires GB10 / SM121")

    # These source members exercise the actual largest FP4 fused FF owner and
    # its down projection, plus each distinct BF16 dense owner geometry.
    groups = {
        "fp4_ff_up": ("fp4", [f"model.layers.0.feed_forward.w{i}.weight" for i in (1, 3)]),
        "fp4_ff_down": ("fp4", ["model.layers.0.feed_forward.w2.weight"]),
        "bf16_conv_in": ("bf16", ["model.layers.0.conv.in_proj.weight"]),
        "bf16_conv_out": ("bf16", ["model.layers.0.conv.out_proj.weight"]),
        "bf16_attention_qkv": ("bf16", [f"model.layers.2.self_attn.{i}_proj.weight" for i in ("q", "k", "v")]),
    }
    wanted = {name for _, members in groups.values() for name in members}
    shapes = {}
    for shard in sorted(args.model.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in wanted.intersection(handle.keys()):
                if name in shapes:
                    raise ValueError(f"duplicate source tensor: {name}")
                shapes[name] = list(handle.get_slice(name).get_shape())
    if set(shapes) != wanted:
        raise ValueError(f"missing representative source members: {sorted(wanted - set(shapes))}")

    record = {
        "schema": "prismaquant.lfm_mixed_native_preflight.v1",
        "scope": "synthetic operand ABI/shape check; no model encoding or cell qualification",
        "image": args.image, "tessera_commit": args.tessera_commit,
        "torch": torch.__version__, "vllm": vllm.__version__,
        "device": torch.cuda.get_device_name(), "compute_capability": [12, 1],
        "source_model": str(args.model), "source_shapes": shapes,
        "seed": 253, "native_threads": torch.get_num_threads(), "rows": [],
        "passed": False, "runtime_price_claimed": False,
    }
    report = args.out / "result.json"
    try:
        native_ops.require_native_fp4_quant("mixed LFM preflight")
        gs = torch.ones((), dtype=torch.float32, device="cuda")
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.inference_mode(), torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            for label, (family, members) in groups.items():
                dimensions = [shapes[name] for name in members]
                if any(len(s) != 2 or s[1] != dimensions[0][1] for s in dimensions):
                    raise ValueError(f"incompatible source member shapes: {label}")
                n, k = sum(s[0] for s in dimensions), dimensions[0][1]
                weight = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
                if family == "fp4":
                    packed_b, scale_b = native_ops.native_fp4_quant(weight, gs)
                    packed_b = packed_b.view(torch.float4_e2m1fn_x2)
                for m in (1, 64):
                    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                    with torch.profiler.record_function(f"{label}/M{m}"):
                        if family == "fp4":
                            packed_a, scale_a = native_ops.native_fp4_quant(x, gs)
                            y = torch._scaled_mm(
                                packed_a.view(torch.float4_e2m1fn_x2), packed_b.t(),
                                scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
                        else:
                            y = torch.mm(x, weight.t(), out_dtype=torch.float32)
                            # Same epilogue ordering as the resident BF16 route.
                            y = (y * torch.ones(n, device="cuda", dtype=torch.float32)).to(torch.bfloat16)
                    torch.cuda.synchronize()
                    if tuple(y.shape) != (m, n) or not bool(torch.isfinite(y).all()):
                        raise ValueError(f"nonfinite or wrong-shaped native output: {label}, M={m}")
                    record["rows"].append({"owner_geometry": label, "family": family,
                                           "m": m, "n": n, "k": k, "finite": True,
                                           "output_sha256": hashlib.sha256(
                                               y.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()})
                del weight, x, y
        profiler.export_chrome_trace(str(args.out / "profile.json"))
        (args.out / "profile.txt").write_text(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
        record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        record["passed"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "native_invocations": len(record["rows"]), "result": str(report)}))


if __name__ == "__main__":
    main()
