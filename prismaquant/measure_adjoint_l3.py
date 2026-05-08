"""Measure adjoint-sketch L3 costs.

This collector builds the ``prismaquant.adjoint_l3.v1`` artifact consumed by
``adjoint_l3.py``.  For every target quantizable tensor and candidate format it
captures a local module-output perturbation ``e_i(f)`` during the clean forward,
then projects it against the backward adjoint ``g_i``:

    a_i(f, r) = <g_i(r), e_i(f)>

Each calibration sample is one sketch rank by default.  The downstream solver
uses ``0.5 / R * ||sum_i a_i(x_i)||^2`` as a PSD surrogate for propagated KL.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.adjoint_l3 import SCHEMA
from prismaquant.build_rtn_cache import (
    iter_quantizable_tensors,
    stage_multimodal,
)
from prismaquant.kl_fisher import fisher_probe_scalar


@dataclass(frozen=True)
class TargetParam:
    name: str
    module: nn.Module
    attr: str
    n_params: int
    shape: tuple[int, ...]


@dataclass
class _ModuleTargets:
    module: nn.Module
    params: list[TargetParam] = field(default_factory=list)


def _recipe_name(full_name: str) -> str:
    return full_name[:-7] if full_name.endswith(".weight") else full_name


def _target_aliases(name: str) -> tuple[str, ...]:
    aliases = {name}
    if name.endswith(".weight"):
        aliases.add(name[:-7])
    else:
        aliases.add(f"{name}.weight")
    if name.startswith("model."):
        suffix = name[len("model."):]
        aliases.add(f"model.language_model.{suffix}")
        aliases.add(f"model.language_model.{suffix}.weight")
    if name.startswith("model.language_model."):
        suffix = name[len("model.language_model."):]
        aliases.add(f"model.{suffix}")
        aliases.add(f"model.{suffix}.weight")
    return tuple(sorted(aliases))


def _first_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for child in value:
            found = _first_tensor(child)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for child in value.values():
            found = _first_tensor(child)
            if found is not None:
                return found
    return None


def _extract_logits(output) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    tensor = _first_tensor(output)
    if tensor is None:
        raise RuntimeError("model output did not contain logits tensor")
    return tensor


def _clone_call_with_quantized_first_tensor(args, kwargs, spec: fr.FormatSpec):
    if spec.act_bits is None or spec.act_bits >= 16:
        return args, kwargs
    quant_fn = spec.activation_quantize_dequantize
    if args and isinstance(args[0], torch.Tensor):
        args = (quant_fn(args[0]),) + tuple(args[1:])
    if kwargs:
        kwargs = dict(kwargs)
        for key in ("hidden_states", "input", "inputs"):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                kwargs[key] = quant_fn(value)
                break
    return args, kwargs


def _causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    if logits.size(1) < 2 or input_ids.size(1) < 2:
        raise ValueError("CE adjoint mode requires sequence length >= 2")
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )


def _sample_token_windows_from_texts(
    texts: Sequence[str],
    tokenizer,
    n_samples: int,
    seqlen: int,
    *,
    seed: int,
) -> torch.Tensor:
    import random

    rng = random.Random(int(seed))
    order = list(range(len(texts)))
    rng.shuffle(order)
    windows: list[torch.Tensor] = []
    buffer: list[int] = []
    eos = tokenizer.eos_token_id
    for idx in order:
        text = str(texts[idx]).strip()
        if not text:
            continue
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        ).input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if not ids:
            continue
        buffer.extend(int(v) for v in ids)
        if eos is not None:
            buffer.append(int(eos))
        while len(buffer) >= int(seqlen) and len(windows) < int(n_samples):
            max_start = len(buffer) - int(seqlen)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            window = buffer[start:start + int(seqlen)]
            windows.append(torch.tensor(window, dtype=torch.long))
            del buffer[:start + int(seqlen)]
        if len(windows) >= int(n_samples):
            break
    if len(windows) < int(n_samples):
        raise RuntimeError(
            f"only built {len(windows)} calibration windows; "
            f"needed {int(n_samples)}"
        )
    return torch.stack(windows, dim=0)


def load_wikitext_calibration_windowed(
    tokenizer,
    n_samples: int,
    seqlen: int,
    *,
    split: str = "train",
    seed: int = 42,
) -> torch.Tensor:
    """Load small calibration windows without tokenizing the full corpus."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [row["text"] for row in ds if str(row.get("text", "")).strip()]
    del ds
    return _sample_token_windows_from_texts(
        texts,
        tokenizer,
        n_samples,
        seqlen,
        seed=seed,
    )


def _teacher_top1_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.size(1) < 2:
        raise ValueError("teacher-top1 adjoint mode requires sequence length >= 2")
    shift_logits = logits[:, :-1, :].contiguous().float()
    labels = shift_logits.detach().argmax(dim=-1)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
    )


def _median_positive_ratio(pairs: Sequence[tuple[float, float]]) -> float:
    ratios = sorted(
        float(left) / float(right)
        for left, right in pairs
        if float(left) > 0.0 and float(right) > 0.0
    )
    if not ratios:
        return 0.0
    mid = len(ratios) // 2
    if len(ratios) % 2:
        return ratios[mid]
    return 0.5 * (ratios[mid - 1] + ratios[mid])


class _AdjointHookCollector:
    def __init__(
        self,
        module_targets: Sequence[_ModuleTargets],
        specs: Sequence[fr.FormatSpec],
        rank: int,
        *,
        error_device: str = "cpu",
        include_activation_quant: bool = True,
        progress_callback: Callable[[dict], None] | None = None,
    ):
        self.module_targets = list(module_targets)
        self.specs = [spec for spec in specs if spec.name != "BF16"]
        self.rank = int(rank)
        self.error_device = str(error_device)
        self.include_activation_quant = bool(include_activation_quant)
        self.progress_callback = progress_callback
        self.handles = []
        self.current_rank = 0
        self._reentrant = False
        self._module_stacks: dict[int, list[dict[tuple[str, str], torch.Tensor]]] = (
            defaultdict(list)
        )
        self.sketches: dict[tuple[str, str], list[float]] = defaultdict(
            lambda: [0.0 for _ in range(self.rank)]
        )
        self.output_mse: dict[tuple[str, str], list[float]] = defaultdict(list)
        self.errors: list[dict] = []

    def install(self) -> None:
        for module_targets in self.module_targets:
            module = module_targets.module
            self.handles.append(
                module.register_forward_hook(
                    self._make_forward_hook(module_targets),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                module.register_full_backward_hook(
                    self._make_backward_hook(module_targets),
                )
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._module_stacks.clear()

    def _make_forward_hook(self, module_targets: _ModuleTargets):
        def _hook(module, args, kwargs, output):
            if self._reentrant:
                return
            clean = _first_tensor(output)
            if clean is None:
                return
            bucket: dict[tuple[str, str], torch.Tensor] = {}
            clean_detached = clean.detach()
            for target in module_targets.params:
                param = getattr(module, target.attr, None)
                if not isinstance(param, torch.nn.Parameter):
                    continue
                original_data = param.data
                for spec in self.specs:
                    try:
                        q_weight = spec.quantize_dequantize(
                            original_data.detach()
                        ).to(device=original_data.device, dtype=original_data.dtype)
                        call_args, call_kwargs = args, kwargs
                        if self.include_activation_quant:
                            call_args, call_kwargs = (
                                _clone_call_with_quantized_first_tensor(
                                    call_args,
                                    call_kwargs,
                                    spec,
                                )
                            )
                        self._reentrant = True
                        try:
                            param.data = q_weight
                            with torch.no_grad():
                                q_output = module(*call_args, **(call_kwargs or {}))
                        finally:
                            param.data = original_data
                            self._reentrant = False
                        q_tensor = _first_tensor(q_output)
                        if q_tensor is None or tuple(q_tensor.shape) != tuple(clean.shape):
                            raise RuntimeError(
                                "quantized local forward returned incompatible output"
                            )
                        err = (q_tensor.detach() - clean_detached).detach()
                        if self.error_device == "cpu":
                            err = err.to("cpu", dtype=torch.float32)
                        elif self.error_device != "cuda":
                            err = err.to(self.error_device)
                        bucket[(target.name, spec.name)] = err
                    except Exception as exc:
                        self.errors.append({
                            "name": target.name,
                            "format": spec.name,
                            "error": str(exc),
                        })
                        if self.progress_callback is not None:
                            self.progress_callback({
                                "event": "target_format_error",
                                "name": target.name,
                                "format": spec.name,
                                "error": str(exc),
                            })
                    finally:
                        param.data = original_data
                        if "q_weight" in locals():
                            del q_weight
            self._module_stacks[id(module)].append(bucket)

        return _hook

    def _make_backward_hook(self, module_targets: _ModuleTargets):
        def _hook(module, _grad_input, grad_output):
            stack = self._module_stacks.get(id(module))
            if not stack:
                return
            bucket = stack.pop()
            grad = _first_tensor(grad_output)
            if grad is None:
                return
            grad = grad.detach()
            for (name, fmt), err in bucket.items():
                if err.device != grad.device:
                    grad_work = grad.to(err.device, dtype=torch.float32)
                    err_work = err.float()
                else:
                    grad_work = grad.float()
                    err_work = err.float()
                projection = torch.sum(grad_work * err_work)
                value = float(projection.detach().cpu().item())
                self.sketches[(name, fmt)][self.current_rank] += value
                self.output_mse[(name, fmt)].append(
                    float(torch.mean(err_work * err_work).detach().cpu().item())
                )

        return _hook


def _collect_targets(
    model: nn.Module,
    *,
    target_names: set[str] | None = None,
    max_targets: int | None = None,
) -> tuple[list[_ModuleTargets], dict[str, TargetParam], list[str]]:
    accepted_aliases = None
    if target_names is not None:
        accepted_aliases = set()
        for name in target_names:
            accepted_aliases.update(_target_aliases(name))

    by_module: dict[int, _ModuleTargets] = {}
    targets: dict[str, TargetParam] = {}
    skipped: list[str] = []
    for full_name, module, attr in iter_quantizable_tensors(model):
        name = _recipe_name(full_name)
        aliases = set(_target_aliases(name)) | set(_target_aliases(full_name))
        if accepted_aliases is not None and not aliases.intersection(accepted_aliases):
            continue
        param = getattr(module, attr, None)
        if not isinstance(param, torch.nn.Parameter):
            skipped.append(name)
            continue
        shape = tuple(int(v) for v in param.shape)
        target = TargetParam(
            name=name,
            module=module,
            attr=attr,
            n_params=int(param.numel()),
            shape=shape,
        )
        targets[name] = target
        by_module.setdefault(id(module), _ModuleTargets(module=module)).params.append(
            target
        )
        if max_targets is not None and len(targets) >= int(max_targets):
            break
    return list(by_module.values()), targets, skipped


def collect_adjoint_l3(
    model: nn.Module,
    calib_ids: torch.Tensor,
    specs: Sequence[fr.FormatSpec],
    *,
    target_names: set[str] | None = None,
    max_targets: int | None = None,
    direction_mode: str = "ce",
    error_device: str = "cpu",
    include_activation_quant: bool = True,
    diagonal_floor_frac: float = 1.0,
    mse_diagonal_floor_frac: float = 0.0,
    fisher_probes_per_sample: int = 1,
    fisher_seed: int = 0,
    fisher_temperature: float = 1.0,
    fisher_token_scope: str = "last",
    fisher_probe_distribution: str = "gaussian",
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    if not isinstance(calib_ids, torch.Tensor):
        raise TypeError("collect_adjoint_l3 currently expects tensor input_ids")
    if calib_ids.dim() != 2:
        raise ValueError("calib_ids must have shape [samples, seqlen]")
    base_sample_count = int(calib_ids.size(0))
    if str(direction_mode).startswith("fisher-") or str(direction_mode).startswith("kl-fisher"):
        probes_per_sample = max(int(fisher_probes_per_sample), 1)
    else:
        probes_per_sample = 1
    rank = base_sample_count * probes_per_sample
    if rank <= 0:
        raise ValueError("calib_ids must contain at least one sample")

    specs_by_name = {
        fr.canonical_format_name(spec.name): fr.REGISTRY[fr.canonical_format_name(spec.name)]
        for spec in specs
    }
    if "BF16" not in specs_by_name:
        specs = [*specs, fr.get_format("BF16")]
        specs_by_name["BF16"] = fr.get_format("BF16")
    specs = [specs_by_name[name] for name in sorted(specs_by_name)]

    module_targets, targets, skipped = _collect_targets(
        model,
        target_names=target_names,
        max_targets=max_targets,
    )
    if not targets:
        raise ValueError("no quantizable targets matched the collector filters")

    model.eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    device = next(model.parameters()).device
    collector = _AdjointHookCollector(
        module_targets,
        specs,
        rank,
        error_device=error_device,
        include_activation_quant=include_activation_quant,
        progress_callback=progress_callback,
    )
    start = time.time()
    collector.install()
    try:
        rank_idx = 0
        for sample_idx in range(base_sample_count):
            batch = calib_ids[sample_idx:sample_idx + 1].to(device)
            for probe_idx in range(probes_per_sample):
                collector.current_rank = rank_idx
                model.zero_grad(set_to_none=True)
                output = model(batch)
                logits = _extract_logits(output)
                if direction_mode == "ce":
                    objective = _causal_lm_loss(logits, batch)
                elif direction_mode == "teacher-top1":
                    objective = _teacher_top1_loss(logits)
                elif direction_mode in {"fisher-last-token", "kl-fisher-last-token"}:
                    objective = fisher_probe_scalar(
                        logits,
                        seed=int(fisher_seed) + rank_idx,
                        token_scope="last",
                        temperature=float(fisher_temperature),
                        distribution=str(fisher_probe_distribution),
                    )
                elif direction_mode == "kl-fisher":
                    objective = fisher_probe_scalar(
                        logits,
                        seed=int(fisher_seed) + rank_idx,
                        token_scope=str(fisher_token_scope),
                        temperature=float(fisher_temperature),
                        distribution=str(fisher_probe_distribution),
                    )
                else:
                    raise ValueError(f"unknown adjoint direction mode: {direction_mode}")
                objective.backward()
                if progress_callback is not None:
                    progress_callback({
                        "event": "sample_done",
                        "sample_index": rank_idx,
                        "sample_count": rank,
                        "calib_sample_index": sample_idx,
                        "probe_index": probe_idx,
                        "loss": float(objective.detach().cpu().item()),
                    })
                del output, logits, objective
                rank_idx += 1
            del batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        collector.remove()
        model.zero_grad(set_to_none=True)

    measured: dict[tuple[str, str], dict[str, float | list[float]]] = {}
    scale_pairs: list[tuple[float, float]] = []
    for name in sorted(targets):
        for spec in specs:
            if spec.name == "BF16":
                continue
            sketch = collector.sketches.get(
                (name, spec.name),
                [0.0 for _ in range(rank)],
            )
            mse_values = collector.output_mse.get((name, spec.name), [])
            mse = sum(mse_values) / max(len(mse_values), 1)
            self_cost = 0.5 / float(rank) * sum(float(v) * float(v) for v in sketch)
            measured[(name, spec.name)] = {
                "sketch": [float(v) for v in sketch],
                "mse": float(mse),
                "self_cost": float(self_cost),
            }
            scale_pairs.append((float(self_cost), float(mse)))
    mse_floor_scale = _median_positive_ratio(scale_pairs)

    units = {}
    for name, target in sorted(targets.items()):
        formats = {}
        for spec in specs:
            if spec.name == "BF16":
                sketch = [0.0 for _ in range(rank)]
                mse = 0.0
                self_cost = 0.0
                diagonal_cost = 0.0
            else:
                item = measured[(name, spec.name)]
                sketch = list(item["sketch"])
                mse = float(item["mse"])
                self_cost = float(item["self_cost"])
                diagonal_cost = (
                    max(float(diagonal_floor_frac), 0.0) * self_cost
                    + max(float(mse_diagonal_floor_frac), 0.0)
                    * float(mse_floor_scale)
                    * mse
                )
            memory_bytes = spec.memory_bytes_for_shape(target.shape)
            formats[spec.name] = {
                "sketch": [float(v) for v in sketch],
                "diagonal_cost": float(diagonal_cost),
                "adjoint_self_cost": float(self_cost),
                "mse_floor_cost": float(
                    max(float(mse_diagonal_floor_frac), 0.0)
                    * float(mse_floor_scale)
                    * float(mse)
                ),
                "bits_per_param": spec.effective_bits_for_shape(target.shape),
                "memory_bytes": int(memory_bytes),
                "output_delta_mse": float(mse),
            }
        units[name] = {
            "formats": formats,
            "shape": list(target.shape),
            "n_params": int(target.n_params),
        }

    return {
        "schema": SCHEMA,
        "rank": rank,
        "normalization": "0.5/rank",
        "direction_mode": direction_mode,
        "units": units,
        "meta": {
            "target_count": len(units),
            "format_names": [spec.name for spec in specs],
            "calib_samples": base_sample_count,
            "rank": rank,
            "direction_mode": str(direction_mode),
            "fisher_probes_per_sample": probes_per_sample,
            "fisher_seed": int(fisher_seed),
            "fisher_temperature": float(fisher_temperature),
            "fisher_token_scope": (
                "last"
                if direction_mode in {"fisher-last-token", "kl-fisher-last-token"}
                else str(fisher_token_scope)
            ),
            "fisher_probe_distribution": str(fisher_probe_distribution),
            "objective_metric": (
                "teacher_forward_kl_single_point_fisher"
                if (
                    str(direction_mode).startswith("fisher-")
                    or str(direction_mode).startswith("kl-fisher")
                )
                else str(direction_mode)
            ),
            "curvature": (
                "categorical_fisher_psd"
                if (
                    str(direction_mode).startswith("fisher-")
                    or str(direction_mode).startswith("kl-fisher")
                )
                else None
            ),
            "calib_seqlen": int(calib_ids.size(1)),
            "include_activation_quant": include_activation_quant,
            "diagonal_floor_frac": float(diagonal_floor_frac),
            "mse_diagonal_floor_frac": float(mse_diagonal_floor_frac),
            "mse_floor_scale": float(mse_floor_scale),
            "error_device": error_device,
            "elapsed_seconds": time.time() - start,
            "skipped_targets": skipped,
            "target_format_errors": collector.errors[:100],
            "target_format_error_count": len(collector.errors),
        },
    }


def _dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _device_from_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _load_target_names(path: str | None) -> set[str] | None:
    if not path:
        return None
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list):
        return {str(item) for item in raw}
    if isinstance(raw, Mapping) and "assignment" in raw:
        return {str(item) for item in raw["assignment"]}
    if isinstance(raw, Mapping):
        return {str(item) for item in raw}
    raise ValueError(f"unsupported target names JSON shape: {path}")


def _progress_printer(event: dict) -> None:
    kind = event.get("event")
    if kind == "sample_done":
        print(
            "[adjoint-l3] sample "
            f"{int(event['sample_index']) + 1}/{int(event['sample_count'])} "
            f"loss={float(event['loss']):.6g}",
            flush=True,
        )
    elif kind == "target_format_error":
        print(
            "[adjoint-l3] target error "
            f"{event.get('name')} {event.get('format')}: {event.get('error')}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure adjoint-sketch L3 costs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--formats", default="NVFP4,MXFP8_E4M3,BF16")
    parser.add_argument("--n-calib-samples", type=int, default=2)
    parser.add_argument("--calib-seqlen", type=int, default=128)
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--target-names-json", default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument(
        "--direction-mode",
        choices=["ce", "teacher-top1", "fisher-last-token", "kl-fisher", "kl-fisher-last-token"],
        default="ce",
    )
    parser.add_argument(
        "--error-device",
        default="cpu",
        help="Where to store local output deltas before backward projection",
    )
    parser.add_argument("--no-activation-quant", action="store_true")
    parser.add_argument(
        "--diagonal-floor-frac",
        type=float,
        default=1.0,
        help="Fraction of each option's low-rank self cost added as diagonal floor",
    )
    parser.add_argument(
        "--mse-diagonal-floor-frac",
        type=float,
        default=0.0,
        help=(
            "Additional output-MSE diagonal floor, scaled by the artifact's "
            "median self_cost/output_mse ratio"
        ),
    )
    parser.add_argument("--fisher-probes-per-sample", type=int, default=1)
    parser.add_argument("--fisher-seed", type=int, default=0)
    parser.add_argument(
        "--fisher-temperature",
        type=float,
        default=1.0,
        help="Teacher/student softmax temperature for KL-Fisher probes.",
    )
    parser.add_argument(
        "--fisher-token-scope",
        choices=["last", "all", "causal"],
        default="last",
        help=(
            "Token positions used by --direction-mode kl-fisher. "
            "The default 'last' matches assignment KL validation."
        ),
    )
    parser.add_argument(
        "--fisher-probe-distribution",
        choices=["gaussian", "rademacher"],
        default="gaussian",
        help="Zero-mean unit-variance noise distribution for Fisher probes.",
    )
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device_from_arg(args.device)
    dtype = _dtype_from_name(args.dtype)
    staged, cleanup = stage_multimodal(args.model)
    try:
        tokenizer_kwargs = {"trust_remote_code": True}
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": bool(args.local_files_only),
        }
        if args.device_map:
            load_kwargs["device_map"] = args.device_map
        tokenizer_kwargs["local_files_only"] = bool(args.local_files_only)
        tokenizer = AutoTokenizer.from_pretrained(staged, **tokenizer_kwargs)
        calib_ids = load_wikitext_calibration_windowed(
            tokenizer,
            args.n_calib_samples,
            args.calib_seqlen,
            split=args.calib_split,
            seed=args.calib_seed,
        )
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        if not args.device_map:
            model.to(device)
        specs = [fr.get_format(part.strip()) for part in args.formats.split(",") if part.strip()]
        target_names = _load_target_names(args.target_names_json)
        payload = collect_adjoint_l3(
            model,
            calib_ids,
            specs,
            target_names=target_names,
            max_targets=args.max_targets,
            direction_mode=args.direction_mode,
            error_device=args.error_device,
            include_activation_quant=not args.no_activation_quant,
            diagonal_floor_frac=args.diagonal_floor_frac,
            mse_diagonal_floor_frac=args.mse_diagonal_floor_frac,
            fisher_probes_per_sample=args.fisher_probes_per_sample,
            fisher_seed=args.fisher_seed,
            fisher_temperature=args.fisher_temperature,
            fisher_token_scope=args.fisher_token_scope,
            fisher_probe_distribution=args.fisher_probe_distribution,
            progress_callback=_progress_printer,
        )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"[adjoint-l3] wrote {out_path} "
            f"rank={payload['rank']} targets={len(payload['units'])} "
            f"errors={payload['meta']['target_format_error_count']}",
            flush=True,
        )
    finally:
        if cleanup is not None:
            import shutil

            shutil.rmtree(cleanup, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
