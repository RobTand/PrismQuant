"""PB-only arithmetic diagnosis from independently captured native FP8 tensors."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import compressed_tensors
    import compressed_tensors.quantization.lifecycle.forward_helpers as ct_forward
    import torch
    from safetensors.torch import load_file
    from prismaquant import fp8_dynamic

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensors", required=True, type=Path)
    parser.add_argument("--tensors-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if sha(args.tensors) != args.tensors_sha256:
        raise ValueError("native tensor artifact hash differs")
    tensors = load_file(str(args.tensors), device="cuda")
    report = {"schema": "prismaquant.fp8_native_arithmetic_diagnostic.v1",
        "input_artifact_sha256": args.tensors_sha256,
        "environment": {"host": socket.gethostname(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "compressed_tensors": compressed_tensors.__version__, "affinity": sorted(os.sched_getaffinity(0)),
            "pq_fp8_source_sha256": sha(inspect.getfile(fp8_dynamic)),
            "compressed_tensors_forward_sha256": sha(inspect.getfile(ct_forward))},
        "phases": {}}

    def tensor_sha(value):
        return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    for phase in ("prefill", "decode"):
        x = tensors[phase + ".input"]
        reference = tensors[phase + ".expected_qdq"]
        codes = tensors[phase + ".native_codes_bits"]
        codes = codes.view(torch.float8_e4m3fn) if codes.dtype == torch.uint8 else codes
        native_scale = tensors[phase + ".native_scales"]
        if x.ndim != 2 or x.dtype != torch.bfloat16 or native_scale.shape != (x.shape[0], 1):
            raise ValueError("diagnostic needs actual BF16 rows and per-token scales")
        shared = fp8_dynamic.fp8_dynamic_activation_qdq_vllm(x)
        shared_qdq = shared.dequant.to(x.dtype)
        rows = x.float()
        amax = rows.abs().amax(dim=-1, keepdim=True)
        native_qdq = (codes.float() * native_scale).to(x.dtype)
        denominator = torch.tensor(448.0, dtype=torch.float32, device=x.device)
        scale_variants = {"shared_scalar_div": shared.scale,
            "tensor_div": (amax / denominator).clamp_min(1 / (448 * 512)),
            "double_div_cast": (amax.double() / 448.0).float().clamp_min(1 / (448 * 512)),
            "float_reciprocal_multiply": (amax * denominator.reciprocal()).clamp_min(1 / (448 * 512))}
        scale_comparisons = {}
        for name, scale in scale_variants.items():
            candidate = (rows / scale).clamp(-448, 448).to(codes.dtype)
            qdq = (candidate.float() * scale).to(x.dtype)
            scale_comparisons[name] = {"scale_values_differ": int((scale != native_scale).sum()),
                "max_scale_abs": float((scale - native_scale).abs().max()),
                "code_bytes_differ": int((candidate.view(torch.uint8) != codes.view(torch.uint8)).sum()),
                "dequant_values_differ": int((qdq != native_qdq).sum()),
                "max_dequant_abs": float((qdq.float() - native_qdq.float()).abs().max()),
                "scales_sha256": tensor_sha(scale)}
        ratios = {"division": rows / shared.scale,
            "reciprocal_multiply": rows * shared.scale.reciprocal(),
            "max_ratio_multiply": rows * (448.0 / amax.clamp_min(1 / 512)),
            "double_max_ratio": rows.double() * (448.0 / amax.double().clamp_min(1 / 512))}
        variants = {
            "shared_compressed_tensors": shared.quant,
            "divide_shared_scale": ratios["division"].clamp(-448, 448).to(codes.dtype),
            "multiply_reciprocal_shared_scale": ratios["reciprocal_multiply"].clamp(-448, 448).to(codes.dtype),
            "multiply_max_ratio": ratios["max_ratio_multiply"].clamp(-448, 448).to(codes.dtype),
            "divide_native_scale": (rows / native_scale).clamp(-448, 448).to(codes.dtype),
            "multiply_reciprocal_native_scale": (rows * native_scale.reciprocal()).clamp(-448, 448).to(codes.dtype),
            "double_max_ratio": ratios["double_max_ratio"].clamp(-448, 448).float().to(codes.dtype),
        }
        compared = {}
        for name, candidate in variants.items():
            qdq = (candidate.float() * native_scale).to(x.dtype)
            compared[name] = {"code_bytes_differ": int((candidate.view(torch.uint8) != codes.view(torch.uint8)).sum()),
                "dequant_values_differ": int((qdq != native_qdq).sum()),
                "max_dequant_abs": float((qdq.float() - native_qdq.float()).abs().max()),
                "codes_sha256": tensor_sha(candidate)}
        indices = torch.nonzero(shared.quant.view(torch.uint8) != codes.view(torch.uint8))[:64].cpu().tolist()
        examples = []
        for row, col in indices:
            examples.append({"row": row, "column": col, "x": float(x[row, col]), "amax": float(amax[row, 0]),
                "shared_scale": float(shared.scale[row, 0]), "native_scale": float(native_scale[row, 0]),
                "shared_code": float(shared.quant[row, col]), "native_code": float(codes[row, col]),
                **{name: float(value[row, col]) for name, value in ratios.items()}})
        report["phases"][phase] = {"shape": list(x.shape), "input_sha256": tensor_sha(x),
            "shared_reproduces_frozen_reference": torch.equal(shared_qdq, reference),
            "native_scale_values_differ": int((shared.scale != native_scale).sum()),
            "max_scale_abs": float((shared.scale - native_scale).abs().max()),
            "scale_variants": scale_comparisons,
            "shared_reference_sha256": tensor_sha(shared_qdq), "native_qdq_sha256": tensor_sha(native_qdq),
            "variants": compared, "first_code_mismatches": examples}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.out), "sha256": sha(args.out), "phases": {
        phase: {key: value for key, value in result.items() if key != "first_code_mismatches"}
        for phase, result in report["phases"].items()}}, allow_nan=False))


if __name__ == "__main__":
    main()
