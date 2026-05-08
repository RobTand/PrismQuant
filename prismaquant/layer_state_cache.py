"""Layer-boundary hidden-state cache for replaying decoder tails.

Design summary:
1. Coord descent changes one Linear at a time, so all hidden states before
   that Linear's decoder layer are identical for every trial at that depth.
2. This module captures the input hidden state for every decoder layer during
   one baseline forward and stores those tensors in `layer_inputs`.
3. `replay_from(L)` starts from the cached input to layer L, executes only
   layers L..N-1, then applies the model's final norm and lm_head.
4. Capture uses decoder-layer forward pre-hooks, which keeps the cache
   independent of a specific model class as long as the model has a normal
   `model.layers`-style decoder stack.
5. Replay preserves each layer call's non-hidden arguments from the baseline
   forward, such as attention masks, position ids, and position embeddings.
6. Baseline format assignments are applied with temporary quantized weight
   copies and optional activation-quantization hooks from `format_registry`.
7. Per-call `weight_override` tensors win over the baseline assignment and are
   restored with try/finally so the live model is left unchanged.
8. The cache is invalidated explicitly when the baseline assignment or
   calibration batch changes.
9. Intended coord-descent wiring:
      cache = LayerHiddenStateCache(model)
      cache.populate(current_assignment, calib_ids, device=device, dtype=dtype)
      layer_idx = layer_index_for_linear(candidate_name)
      trial_weight = quantized_weight_for(candidate_name, candidate_fmt)
      logits = cache.replay_from(layer_idx, {candidate_name: trial_weight})
      # compute KL/log-prob loss from logits, then either keep or reject trial
      cache.invalidate()  # after committing a baseline assignment change
10. The module does not call coord descent directly; follow-up integration only
    needs to route each trial to the changed Linear's decoder layer.
"""
from __future__ import annotations

import inspect
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.perturbed_x_cache import _maybe_clip_activations


_HIDDEN_SENTINEL = object()


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class _TargetKey:
    module_id: int
    attr: str


@dataclass
class _WeightTarget:
    key: _TargetKey
    module: nn.Module
    attr: str
    names: tuple[str, ...]


@dataclass
class _LayerCallTemplate:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    hidden_where: str
    hidden_key: int | str


def _as_sequence(value: Any) -> Sequence[nn.Module]:
    if isinstance(value, (nn.ModuleList, list, tuple)):
        return value
    raise TypeError(f"decoder layers must be a sequence, got {type(value).__name__}")


def _resolve_attr_path(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, (list, tuple, nn.ModuleList)) and part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _resolve_parent_path(root: Any, path: str) -> tuple[Any, str]:
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise ValueError("decoder_layers_attr must not be empty")
    parent = root if len(parts) == 1 else _resolve_attr_path(root, ".".join(parts[:-1]))
    return parent, parts[-1]


def _discover_layers(model: nn.Module, preferred_path: str) -> tuple[str, Any, Sequence[nn.Module]]:
    candidate_paths = [
        preferred_path,
        "model.layers",
        "language_model.model.layers",
        "model.decoder.layers",
        "decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]
    seen: set[str] = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            parent, attr = _resolve_parent_path(model, path)
            layers = _as_sequence(getattr(parent, attr))
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        if len(layers) > 0:
            return path, parent, layers

    suffixes = ("layers", "h", "blocks")
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) == 0:
            continue
        if name.rsplit(".", 1)[-1] not in suffixes:
            continue
        parent_path, _, attr = name.rpartition(".")
        parent = model if not parent_path else _resolve_attr_path(model, parent_path)
        return name, parent, module

    raise AttributeError(
        f"could not discover decoder layers from {preferred_path!r}; "
        "pass decoder_layers_attr explicitly"
    )


def _add_language_model_aliases(names: set[str]) -> None:
    for name in list(names):
        if name.startswith("model."):
            suffix = name[len("model."):]
            names.add(f"model.language_model.{suffix}")
        if name.startswith("language_model.model."):
            suffix = name[len("language_model."):]
            names.add(suffix)


def _build_linear_weight_targets(model: nn.Module) -> tuple[dict[str, _WeightTarget], dict[_TargetKey, _WeightTarget]]:
    by_name: dict[str, _WeightTarget] = {}
    by_key: dict[_TargetKey, _WeightTarget] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        names = {module_name, f"{module_name}.weight"} if module_name else {"weight"}
        _add_language_model_aliases(names)
        key = _TargetKey(id(module), "weight")
        target = _WeightTarget(key=key, module=module, attr="weight", names=tuple(sorted(names)))
        by_key[key] = target
        for name in names:
            by_name[name] = target
    return by_name, by_key


def _first_tensor_location(args: tuple[Any, ...], kwargs: Mapping[str, Any] | None):
    for idx, value in enumerate(args):
        if isinstance(value, torch.Tensor):
            return "args", idx, value
    if kwargs:
        for key in ("hidden_states", "inputs_embeds", "input"):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                return "kwargs", key, value
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                return "kwargs", key, value
    return None, None, None


def _replace_tensor_input(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None,
    where: str | None,
    key: int | str | None,
    value: torch.Tensor,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    out_kwargs = dict(kwargs or {})
    if where == "args":
        out_args = list(args)
        out_args[int(key)] = value
        return tuple(out_args), out_kwargs
    if where == "kwargs":
        out_kwargs[str(key)] = value
        return args, out_kwargs
    return args, out_kwargs


def _detach_template_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_template_value(v) for v in value)
    if isinstance(value, list):
        return [_detach_template_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _detach_template_value(v) for k, v in value.items()}
    return value


def _make_layer_template(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None,
    where: str,
    key: int | str,
) -> _LayerCallTemplate:
    template_args = [_detach_template_value(v) for v in args]
    template_kwargs = {k: _detach_template_value(v) for k, v in dict(kwargs or {}).items()}
    if where == "args":
        template_args[int(key)] = _HIDDEN_SENTINEL
    elif where == "kwargs":
        template_kwargs[str(key)] = _HIDDEN_SENTINEL
    else:
        raise RuntimeError("decoder layer call did not expose a tensor input")
    return _LayerCallTemplate(
        args=tuple(template_args),
        kwargs=template_kwargs,
        hidden_where=where,
        hidden_key=key,
    )


def _fill_template(template: _LayerCallTemplate, hidden: torch.Tensor) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args = list(template.args)
    kwargs = dict(template.kwargs)
    if template.hidden_where == "args":
        args[int(template.hidden_key)] = hidden
    elif template.hidden_where == "kwargs":
        kwargs[str(template.hidden_key)] = hidden
    else:
        raise RuntimeError(f"unknown hidden location {template.hidden_where!r}")
    return tuple(args), kwargs


def _call_accepts_kwarg(fn: Any, name: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters


def _disable_decoder_cache_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(kwargs or {})
    if "use_cache" in out:
        out["use_cache"] = False
    if "past_key_value" in out:
        out["past_key_value"] = None
    if "past_key_values" in out:
        out["past_key_values"] = None
    return out


def _hidden_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, Mapping):
        for key in ("hidden_states", "last_hidden_state"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    for attr in ("hidden_states", "last_hidden_state"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"decoder layer returned no hidden-state tensor: {type(output).__name__}")


class LayerHiddenStateCache:
    """Cache decoder-layer input hidden states and replay only a decoder tail."""

    def __init__(self, model: nn.Module, decoder_layers_attr: str = "model.layers"):
        self.model = model
        self.decoder_layers_attr, self.layers_parent, layers = _discover_layers(
            model,
            decoder_layers_attr,
        )
        self.layers = list(layers)
        self.layer_inputs: list[torch.Tensor] = []
        self.baseline_assignment: dict[str, str] = {}
        self.missing_baseline_names: list[str] = []
        self.skipped_activation_quant: list[dict[str, Any]] = []
        self._layer_call_templates: list[_LayerCallTemplate] = []
        self._linear_targets_by_name, self._linear_targets_by_key = _build_linear_weight_targets(model)
        self._baseline_weight_values: dict[_TargetKey, tuple[_WeightTarget, torch.Tensor]] = {}
        self._activation_quantizers: dict[
            int, tuple[nn.Module, fr.FormatSpec, str]
        ] = {}
        self._activation_max_abs: dict[str, float] = {}
        self._production_weight_cache = None
        self.include_activation_quant = True
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

    def populate(
        self,
        baseline_assignment: Mapping[str, str],
        calib_ids: torch.Tensor,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        include_activation_quant: bool = True,
        production_weight_cache=None,
    ) -> None:
        """Run a baseline forward and cache the input of each decoder layer."""
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("LayerHiddenStateCache requested CUDA, but CUDA is not available")

        self.invalidate()
        self._device = torch_device
        self._dtype = dtype
        self.include_activation_quant = bool(include_activation_quant)
        self._production_weight_cache = production_weight_cache
        self._activation_max_abs = dict(
            getattr(production_weight_cache, "activation_max_abs", None)
            or getattr(production_weight_cache, "activation_scales", None)
            or {}
        )
        self.baseline_assignment = {str(k): str(v) for k, v in baseline_assignment.items()}
        self.model.to(device=torch_device, dtype=dtype)
        self.model.eval()
        self._linear_targets_by_name, self._linear_targets_by_key = _build_linear_weight_targets(self.model)
        self._prepare_baseline_execution()

        captured_inputs: list[torch.Tensor | None] = [None] * len(self.layers)
        captured_templates: list[_LayerCallTemplate | None] = [None] * len(self.layers)
        capture_errors: list[str] = []

        def make_hook(layer_idx: int):
            def _hook(_module, args, kwargs):
                layer_kwargs = _disable_decoder_cache_kwargs(kwargs)
                where, key, hidden = _first_tensor_location(tuple(args), layer_kwargs)
                if not isinstance(hidden, torch.Tensor):
                    capture_errors.append(f"layer {layer_idx} did not receive a tensor hidden state")
                    return None
                captured_inputs[layer_idx] = hidden.detach().clone().contiguous()
                captured_templates[layer_idx] = _make_layer_template(
                    tuple(args),
                    layer_kwargs,
                    str(where),
                    key,
                )
                return None

            return _hook

        handles = [
            layer.register_forward_pre_hook(make_hook(idx), with_kwargs=True)
            for idx, layer in enumerate(self.layers)
        ]
        try:
            with torch.no_grad(), self._temporary_execution():
                call_kwargs = {}
                if _call_accepts_kwarg(self.model.forward, "use_cache"):
                    call_kwargs["use_cache"] = False
                self.model(calib_ids.to(torch_device), **call_kwargs)
        finally:
            for handle in handles:
                handle.remove()

        if capture_errors:
            raise RuntimeError("; ".join(capture_errors))
        missing = [str(idx) for idx, value in enumerate(captured_inputs) if value is None]
        if missing:
            raise RuntimeError(f"failed to capture decoder layer inputs for layers: {', '.join(missing)}")
        template_missing = [str(idx) for idx, value in enumerate(captured_templates) if value is None]
        if template_missing:
            raise RuntimeError(f"failed to capture decoder layer call templates for layers: {', '.join(template_missing)}")

        self.layer_inputs = [value for value in captured_inputs if value is not None]
        self._layer_call_templates = [
            value for value in captured_templates if value is not None
        ]

    def replay_from(
        self,
        layer_idx: int,
        weight_override: Mapping[str, torch.Tensor] | None = None,
        *,
        return_logits: bool = True,
        last_token_only: bool = False,
    ) -> torch.Tensor:
        """Replay from cached decoder-layer input through the model tail."""
        self._require_populated()
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise IndexError(
                f"layer_idx must be in [0, {len(self.layers) - 1}], got {layer_idx}"
            )
        if self._device is None:
            raise RuntimeError("cache has no execution device; call populate first")

        hidden = self.layer_inputs[layer_idx].to(self._device).clone()
        with torch.no_grad(), self._temporary_execution(weight_override):
            for idx in range(layer_idx, len(self.layers)):
                args, kwargs = _fill_template(self._layer_call_templates[idx], hidden)
                kwargs = _disable_decoder_cache_kwargs(kwargs)
                output = self.layers[idx](*args, **kwargs)
                hidden = _hidden_from_layer_output(output)
            hidden = self._apply_final_norm(hidden)
            if not return_logits:
                return hidden
            if last_token_only and hidden.dim() >= 3:
                hidden = hidden[:, -1:, :]
            return self._apply_lm_head(hidden)

    def invalidate(self) -> None:
        """Clear cached states after changing the baseline assignment."""
        self.layer_inputs = []
        self._layer_call_templates = []
        self._baseline_weight_values = {}
        self._activation_quantizers = {}
        self.missing_baseline_names = []
        self.skipped_activation_quant = []

    def cache_nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.layer_inputs)

    def _require_populated(self) -> None:
        if not self.layer_inputs or not self._layer_call_templates:
            raise RuntimeError(
                "LayerHiddenStateCache is empty; call populate(...) before replay_from(...)"
            )

    def _prepare_baseline_execution(self) -> None:
        specs_by_target: dict[_TargetKey, tuple[_WeightTarget, fr.FormatSpec, str]] = {}
        self.missing_baseline_names = []
        self.skipped_activation_quant = []
        external_weight_management = _env_truthy(
            "PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT",
            default=False,
        )
        for name, fmt in self.baseline_assignment.items():
            target = self._linear_targets_by_name.get(name)
            if target is None:
                self.missing_baseline_names.append(name)
                continue
            spec = fr.get_format(fmt)
            existing = specs_by_target.get(target.key)
            if existing is not None and existing[1].name != spec.name:
                names = ", ".join(target.names)
                raise ValueError(f"conflicting baseline formats for shared Linear {names}")
            specs_by_target[target.key] = (target, spec, name)

        self._baseline_weight_values = {}
        activation_specs: dict[int, tuple[nn.Module, fr.FormatSpec, list[str]]] = {}
        activation_conflicts: set[int] = set()
        for key, (target, spec, assignment_name) in specs_by_target.items():
            param = getattr(target.module, target.attr)
            if not isinstance(param, torch.nn.Parameter):
                continue
            canonical = fr.canonical_format_name(spec.name)
            if not external_weight_management:
                production = (
                    self._production_weight_cache.get(assignment_name, canonical)
                    if self._production_weight_cache is not None
                    else None
                )
                if production is not None:
                    quantized = production.to(
                        device=param.device,
                        dtype=param.dtype,
                    ).contiguous()
                else:
                    if (
                        self._production_weight_cache is not None
                        and canonical not in {"BF16", "MXFP8", "MXFP8_E4M3"}
                        and _env_truthy("PRISMAQUANT_STRICT_PRODUCTION_CACHE")
                    ):
                        raise RuntimeError(
                            f"production_weight_cache miss for "
                            f"({assignment_name!r}, {canonical!r}); set "
                            "PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 to fall back "
                            "to RTN, or rebuild the production cache."
                        )
                    quantized = spec.quantize_dequantize(
                        param.data.detach().clone()
                    ).to(
                        device=param.device,
                        dtype=param.dtype,
                    ).contiguous()
                self._baseline_weight_values[key] = (target, quantized)
            if (
                self.include_activation_quant
                and spec.act_bits is not None
                and spec.act_bits < 16
            ):
                if id(target.module) in activation_conflicts:
                    continue
                entry = activation_specs.get(id(target.module))
                if entry is None:
                    activation_specs[id(target.module)] = (
                        target.module, spec, [assignment_name]
                    )
                elif entry[1].name == spec.name:
                    entry[2].append(assignment_name)
                else:
                    self.skipped_activation_quant.append(
                        {
                            "module": type(target.module).__name__,
                            "weights": sorted([*entry[2], assignment_name]),
                            "formats": sorted({entry[1].name, spec.name}),
                        }
                    )
                    activation_specs.pop(id(target.module), None)
                    activation_conflicts.add(id(target.module))
        self._activation_quantizers = {
            module_id: (module, spec, names[0])
            for module_id, (module, spec, names) in activation_specs.items()
        }

    @contextmanager
    def _temporary_execution(
        self,
        weight_override: Mapping[str, torch.Tensor] | None = None,
    ) -> Iterator[None]:
        handles = [
            module.register_forward_pre_hook(
                self._make_activation_quant_hook(
                    spec,
                    name,
                    self._activation_max_abs,
                ),
                with_kwargs=True,
            )
            for module, spec, name in self._activation_quantizers.values()
        ]
        originals: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            if _env_truthy(
                "PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT",
                default=False,
            ):
                values = {}
            else:
                values = dict(self._baseline_weight_values)
            for name, tensor in (weight_override or {}).items():
                target = self._linear_targets_by_name.get(str(name))
                if target is None:
                    raise KeyError(f"weight_override target {name!r} is not an nn.Linear weight")
                param = getattr(target.module, target.attr)
                if tuple(tensor.shape) != tuple(param.shape):
                    raise ValueError(
                        f"weight_override for {name!r} has shape {tuple(tensor.shape)}, "
                        f"expected {tuple(param.shape)}"
                    )
                values[target.key] = (target, tensor.detach())

            for target, tensor in values.values():
                param = getattr(target.module, target.attr)
                if not isinstance(param, torch.nn.Parameter):
                    continue
                originals.append((param, param.data.detach().clone()))
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
            yield
        finally:
            for param, original in reversed(originals):
                param.data.copy_(original.to(device=param.device, dtype=param.dtype))
            for handle in handles:
                handle.remove()

    @staticmethod
    def _make_activation_quant_hook(
        spec: fr.FormatSpec,
        name: str,
        activation_max_abs: Mapping[str, float],
    ):
        def _hook(_module, args, kwargs):
            where, key, hidden = _first_tensor_location(tuple(args), kwargs)
            if not isinstance(hidden, torch.Tensor):
                return None
            hidden = _maybe_clip_activations(hidden, activation_max_abs, name)
            quantized = spec.activation_quantize_dequantize(hidden)
            return _replace_tensor_input(tuple(args), kwargs, where, key, quantized)

        return _hook

    def _apply_final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        for owner in (self.layers_parent, self.model):
            for attr in ("norm", "ln_f", "final_layernorm", "final_layer_norm", "post_norm"):
                module = getattr(owner, attr, None)
                if isinstance(module, nn.Module):
                    return module(hidden)
        return hidden

    def _apply_lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        for owner in (self.model, self.layers_parent):
            module = getattr(owner, "lm_head", None)
            if isinstance(module, nn.Module):
                return module(hidden)
        return hidden
