"""Build activation caches under a perturbed allocation.

The regular probe cache captures BF16 model inputs. Perturbed-X iterations need
the same cache shape after upstream layers have already run with the current
allocation's weight and activation quantization. This module installs one
forward_pre_hook per quantized module: it snapshots the original input first,
then returns the activation-quantized input for the actual forward. Weights are
RTN-quantized just for that module call and restored in the forward hook.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import iter_quantizable_tensors
from prismaquant.memory_management import (
    enforce_gpu_memory_budget,
    env_int,
    env_truthy as _env_truthy,
    model_device as _model_device,
    register_budget_evictor,
)
from prismaquant.nvfp4_activation_contract import (
    require_matching_input_global_scale,
)

_FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")
_SHARED_FROZEN_WEIGHT_FORMAT_CACHE: OrderedDict[
    tuple[str, str, int, str, str],
    torch.Tensor,
] = OrderedDict()


# Clamping inputs to the calibrated max(|activations|) before per-group
# RTN matches the export's act-clip behavior.  Without this, dynamic
# per-group RTN sets scales from the raw input's local max, so outliers
# dominate and any pre-scaling is mathematically a no-op
# (Q(x/s)*s == Q(x) under purely dynamic Q — codex round-3 caught this).


def _activation_max_abs_lookup(
    activation_max_abs: dict,
    param_name: str | None,
) -> float | None:
    """Resolve ``param_name`` against ``activation_max_abs`` with the same
    alias-fallbacks as ``ProductionWeightCache.get`` so cache hits and
    activation-clip lookups stay consistent."""
    if param_name is None or not activation_max_abs:
        return None
    candidates = [param_name]
    if param_name.endswith(".weight"):
        candidates.append(param_name[:-len(".weight")])
    if param_name.startswith("model.language_model."):
        candidates.append("model." + param_name[len("model.language_model."):])
    for cand in candidates:
        v = activation_max_abs.get(cand)
        if v is not None:
            return v
    return None


def _maybe_clip_activations(
    x: "torch.Tensor",
    activation_max_abs: dict,
    param_name: str | None,
) -> "torch.Tensor":
    """Clamp activations to ±max_abs when a calibrated value is known.

    ``activation_max_abs`` is the dict from
    ``ProductionWeightCache.activation_max_abs`` (calibrated max(|x|)
    per fused-sibling group).  Returns ``x`` unchanged when:

      * no entry is registered for ``param_name`` (or its aliases),
      * the registered value is non-positive, or
      * ``PRISMAQUANT_PROD_ACT_SCALES`` is explicitly disabled.
    """
    max_abs = _activation_max_abs_lookup(activation_max_abs, param_name)
    if max_abs is None or max_abs <= 0:
        return x
    if not _env_truthy("PRISMAQUANT_PROD_ACT_SCALES", default=True):
        return x
    return x.clamp(-float(max_abs), float(max_abs))



def _served_nvfp4_act_qdq_enabled() -> bool:
    """Opt-in serve-faithful NVFP4 activation emulation (default OFF).

    When on, activation quantization in the emulation hooks for a spec whose
    served contract is static-scale (``FormatSpec.static_activation_contract``,
    i.e. stock NVFP4) models the SERVED two-level semantics (static
    input_global_scale + FP8 snap of the per-16-group block scale, via the
    contract's own oracle) instead of the dynamic exact-fp32-scale RTN.
    Closes the M18-residual/C1 measurement gap the 2026-07-02 audit flagged;
    default-off pending a served correlation study (the dynamic path is the
    long-standing screen baseline).  A spec whose contract says
    ``measured_as_served`` (a Tessera W4A4 rung) does not consult this lever:
    the served oracle is its only measurement."""
    return os.environ.get(
        "PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES", "0") == "1"


def _activation_qdq(
    x: torch.Tensor,
    act_spec,
    activation_max_abs: dict,
    param_name: str | None,
    priced_input_global_scales: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Shared activation quantize-dequantize for the emulation hooks.

    Which quantizer a spec serves is the SPEC's answer
    (``FormatSpec.static_activation_contract``), never a compare of its name
    against ``"NVFP4"`` -- a Tessera rung routed through the same kernel has
    the same contract and a different name (#205).

    * No static contract (FP8/MX dynamic W8A8, or an A16 row that reached
      the hook): act-clip to the calibrated max_abs, then the row's own
      dynamic quantizer.
    * Static contract, ``measured_as_served`` (Tessera W4A4): the served
      oracle at the unit's G -- NO clamp (serving does not clamp; the static
      scale itself clips blocks above the calibration amax) -- and a refusal
      by name when the unit has no calibrated maximum.
    * Static contract, screen default (stock NVFP4): the historical clip +
      dynamic RTN, or the served oracle when
      ``PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES=1`` and the maximum is
      known.

    ``priced_input_global_scales`` is the G each unit's cached render score was
    priced at (``ProductionWeightCache`` ``render_scores`` provenance, via
    ``production_cache_priced_input_global_scales``).  When one is known it is
    compared against the G this hook is about to apply, and a disagreement
    refuses by name: measuring an assignment whose costs were priced under one
    activation-scale policy through a hook quantizing under another compares
    two different quantizers (#227).  Absent -- no production cache, or a cache
    with no served rows -- there is nothing to disagree with and the hook is
    unchanged."""
    contract = getattr(act_spec, "static_activation_contract", None)
    if contract is not None:
        max_abs = _activation_max_abs_lookup(activation_max_abs, param_name)
        if contract.measured_as_served:
            g = contract.require_input_global_scale(
                max_abs, qname=param_name, consumer="assignment-KL hook")
            g = require_matching_input_global_scale(
                _activation_max_abs_lookup(
                    priced_input_global_scales or {}, param_name),
                g,
                qname=param_name,
                consumer="assignment-KL hook",
            )
            return contract.quantize_dequantize(x, g)
        if (
            _served_nvfp4_act_qdq_enabled()
            and x.shape[-1] % int(contract.group_size) == 0
            and max_abs is not None and max_abs > 0
        ):
            g = contract.input_global_scale_from_max_abs(float(max_abs))
            return contract.quantize_dequantize(x, g)
    x = _maybe_clip_activations(x, activation_max_abs, param_name)
    return act_spec.activation_quantize_dequantize(x)


def activation_cache_filename(name: str) -> str:
    return _FNAME_SUB.sub("__", name) + ".pt"


def write_activation_cache_entry(cache_dir, name, inputs, *, source="perturbed_x",
                                 durable=False, **metadata):
    """Atomically store already-selected rows without changing their precision."""
    import os
    path = Path(cache_dir) / activation_cache_filename(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("wb") as handle:
        torch.save({**metadata, "inputs": inputs.contiguous(), "name": name,
                    "source": source}, handle)
        if durable:
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(temporary, path)
    if durable:
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return path


def _tensor_hash_update(h: "hashlib._Hash", tensor: torch.Tensor) -> None:
    t = tensor.detach().to("cpu").contiguous()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.view(torch.uint8).numpy().tobytes())


def calibration_data_hash(calibration_data) -> str:
    """Stable content hash used to seed shared row subsampling."""
    h = hashlib.blake2b(digest_size=16)
    if isinstance(calibration_data, torch.Tensor):
        _tensor_hash_update(h, calibration_data)
        return h.hexdigest()
    if isinstance(calibration_data, Mapping):
        for key in sorted(calibration_data):
            h.update(str(key).encode())
            value = calibration_data[key]
            if isinstance(value, torch.Tensor):
                _tensor_hash_update(h, value)
            else:
                h.update(repr(value).encode())
        return h.hexdigest()
    for sample in calibration_data:
        if isinstance(sample, torch.Tensor):
            _tensor_hash_update(h, sample)
        elif isinstance(sample, Mapping):
            for key in sorted(sample):
                h.update(str(key).encode())
                value = sample[key]
                if isinstance(value, torch.Tensor):
                    _tensor_hash_update(h, value)
                else:
                    h.update(repr(value).encode())
        else:
            h.update(repr(sample).encode())
    return h.hexdigest()


def _seed_from(cal_hash: str, group_key: str) -> int:
    digest = hashlib.blake2b(
        f"{cal_hash}:{group_key}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def fused_subsample_group(name: str, profile=None) -> str:
    """Return the deterministic row-subsample group for a recipe name."""
    if profile is not None:
        try:
            group = profile.fused_sibling_group(name)
            if group is not None:
                return str(group)
        except Exception:
            pass
    bare = name[:-7] if name.endswith(".weight") else name
    parent, _, leaf = bare.rpartition(".")
    if leaf in {"q_proj", "k_proj", "v_proj"}:
        return f"{parent.rsplit('.', 1)[0]}.qkv"
    if leaf in {"gate_proj", "up_proj"}:
        return f"{parent.rsplit('.', 1)[0]}.gate_up"
    if leaf in {"in_proj_qkv", "in_proj_z"}:
        return f"{parent.rsplit('.', 1)[0]}.in_proj_qkvz"
    if leaf in {"in_proj_a", "in_proj_b"}:
        return f"{parent.rsplit('.', 1)[0]}.in_proj_ab"
    return bare


class SharedRowSubsampler:
    """Deterministic, sibling-coherent row sampling for activation capture.

    Fused siblings (q/k/v, gate/up) must snapshot the SAME rows so their
    caches stay row-aligned for joint solvers.  ``batch_priorities``
    returns the per-row random reservoir priorities for one capture
    batch, keyed by the fused-sibling *group* and the batch index —
    every sibling observing the same batch therefore draws identical
    priorities, and the reservoir's keep/replace decisions match
    row-for-row across the whole calibration stream (the cross-batch
    analogue of the historical shared per-call ``randperm``).

    Seeding follows the existing convention: ``_seed_from(cal_hash, key)``
    so a fixed calibration set reproduces the same sample run-to-run."""

    def __init__(self, input_rows: int, cal_hash: str, profile=None):
        self.input_rows = int(input_rows)
        self.cal_hash = cal_hash
        self.profile = profile

    def batch_priorities(
        self,
        name: str,
        batch_index: int,
        n_rows: int,
    ) -> torch.Tensor:
        """Random priorities for the ``batch_index``-th capture of ``name``.

        Regenerated deterministically per (group, batch) instead of cached:
        zero retained state, and siblings that consume the same batch at
        different times still agree exactly."""
        group = fused_subsample_group(name, self.profile)
        g = torch.Generator(device="cpu")
        g.manual_seed(
            _seed_from(self.cal_hash, f"{group}#batch{int(batch_index)}")
        )
        return torch.rand(int(n_rows), generator=g, dtype=torch.float32)


@dataclass
class _ParamPlan:
    name: str
    attr: str
    spec: fr.FormatSpec


@dataclass
class _ModulePlan:
    module: nn.Module
    params: list[_ParamPlan] = field(default_factory=list)
    active_originals: list[tuple[torch.nn.Parameter, torch.Tensor]] = field(
        default_factory=list
    )
    act_spec: fr.FormatSpec | None = None
    act_conflict: bool = False

    @property
    def cache_names(self) -> list[str]:
        return [p.name for p in self.params]


def _module_input_member_name(plan: _ModulePlan, x: torch.Tensor) -> str | None:
    """Pick the plan member whose calibrated scale describes ``x``.

    Dense Linears have one ``weight`` param — trivially it. A packed-MoE
    experts module carries one plan with SEVERAL per-projection params
    (``gate_up_proj`` + ``down_proj``); their calibrated ``max_abs`` were
    measured on DIFFERENT tensors (module input vs routed post-SwiGLU
    intermediates), and this hook only ever sees the module input, so the
    clip scale must come from the projection that consumes it. Select
    structurally: the param whose in-dim (last weight axis, [E, out, in])
    equals ``x``'s feature dim. Falling back to ``params[0]`` (the old
    behavior) made the clip depend on assignment-dict ordering.
    """
    member_name = next(
        (p.name for p in plan.params if p.attr == "weight"), None)
    if member_name is not None or not plan.params:
        return member_name
    in_dim = int(x.size(-1))
    matches = []
    for p in plan.params:
        tensor = getattr(plan.module, p.attr, None)
        shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[-1]) == in_dim:
            matches.append(p)
    if not matches:
        return plan.params[0].name
    if len(matches) > 1:
        # Degenerate square case (intermediate == hidden): the shape test
        # cannot separate the projections, so break the tie by role — the
        # down projection (down_proj / w2) consumes the internal
        # intermediate, never the module input.
        non_down = [
            p for p in matches
            if p.attr.rsplit(".", 1)[-1] not in ("down_proj", "w2")
        ]
        if non_down:
            return non_down[0].name
    return matches[0].name


def build_quantizable_map(
    model: nn.Module,
    profile=None,
) -> dict[str, tuple[nn.Module, str]]:
    """Map recipe/probe names to live module parameters."""
    out: dict[str, tuple[nn.Module, str]] = {}
    for full_name, mod, attr in iter_quantizable_tensors(model, profile):
        names = {full_name}
        if full_name.endswith(".weight"):
            names.add(full_name[:-7])
        if profile is not None:
            qname = (
                full_name[:-7]
                if attr == "weight" and full_name.endswith(".weight")
                else full_name
            )
            try:
                recipe_name = profile.live_to_recipe_name(qname)
            except Exception:
                recipe_name = qname
            names.add(recipe_name)
            if attr == "weight":
                names.add(f"{recipe_name}.{attr}")
        for name in list(names):
            if name.startswith("model."):
                suffix = name[len("model."):]
                names.add(f"model.language_model.{suffix}")
        for name in names:
            out[name] = (mod, attr)
    return out


def _build_module_plans(
    model: nn.Module,
    assignment: Mapping[str, str],
    profile=None,
) -> tuple[list[_ModulePlan], list[str], list[dict]]:
    quant_map = build_quantizable_map(model, profile=profile)
    by_module: dict[int, _ModulePlan] = {}
    missing: list[str] = []
    for name, fmt in assignment.items():
        target = quant_map.get(name)
        if target is None:
            missing.append(name)
            continue
        mod, attr = target
        spec = fr.get_format(fmt)
        plan = by_module.setdefault(id(mod), _ModulePlan(module=mod))
        plan.params.append(_ParamPlan(name=name, attr=attr, spec=spec))

    skipped: list[dict] = []
    for plan in by_module.values():
        low_act = {
            p.spec.name: p.spec
            for p in plan.params
            if p.spec.act_quant_changes_input
        }
        if len(low_act) == 1:
            plan.act_spec = next(iter(low_act.values()))
        elif len(low_act) > 1:
            plan.act_conflict = True
            skipped.append(
                {
                    "module": type(plan.module).__name__,
                    "weights": sorted(plan.cache_names),
                    "formats": sorted(low_act),
                }
            )
    return list(by_module.values()), missing, skipped


def _first_tensor_location(args, kwargs):
    if args:
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


def _replace_tensor_input(args, kwargs, where, key, value):
    if where == "args":
        args_list = list(args)
        args_list[int(key)] = value
        return tuple(args_list), kwargs
    if where == "kwargs":
        kwargs = dict(kwargs or {})
        kwargs[key] = value
        return args, kwargs
    return args, kwargs


class PerturbedActivationCache:
    def __init__(
        self,
        model: nn.Module,
        assignment: Mapping[str, str],
        cache_dir: str | Path,
        *,
        input_rows: int = 256,
        cal_hash: str,
        profile=None,
        production_weight_cache=None,
        include_activation_quant: bool = True,
        capture_inputs: bool = True,
    ):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.input_rows = int(input_rows)
        self.include_activation_quant = bool(include_activation_quant)
        self.capture_inputs = bool(capture_inputs)
        self.subsampler = SharedRowSubsampler(input_rows, cal_hash, profile)
        self.plans, self.missing, self.skipped = _build_module_plans(
            model, assignment, profile=profile
        )
        self._production_weight_cache = production_weight_cache
        # MED-3: per-Linear calibrated max(|activations|), unified across
        # fused-sibling groups.  Used by the activation-quant hook to
        # clamp activations to ±max_abs before per-group RTN, matching
        # the export's act-clip behavior.  See production_weight_cache.py
        # for the convention note (we store max_abs directly; the export's
        # vLLM-facing metadata is derived via
        # export_native_compressed._nvfp4_input_global_scale_from_max_abs).
        if production_weight_cache is not None and (
            production_weight_cache.activation_max_abs
            or production_weight_cache.activation_scales
        ):
            src = (
                production_weight_cache.activation_max_abs
                or production_weight_cache.activation_scales
            )
            self._activation_scales: dict[str, float] = dict(src)
        else:
            self._activation_scales = {}
        # #227: the maximum above says what the unit's activations reach; it
        # does NOT say which input-global-scale policy the cache's costs were
        # priced under, and the same maximum prices G=6/amax or G=448*6/amax.
        # The cache's own render-score provenance does say, so carry the G each
        # unit was priced at and let the hook refuse by name when the G it
        # would apply is a different one.
        self._priced_input_global_scales: dict[str, float] = {}
        if production_weight_cache is not None:
            from prismaquant.production_weight_cache import (
                production_cache_priced_input_global_scales,
            )

            self._priced_input_global_scales = (
                production_cache_priced_input_global_scales(
                    production_weight_cache,
                    where="assignment-KL hooks",
                )
            )
        # Bounded uniform row reservoirs (M8): per name, at most
        # `input_rows` CPU rows + their float32 priorities, plus a batch
        # counter that keys the shared per-batch priorities.
        self._snap_rows: dict[str, torch.Tensor] = {}
        self._snap_priorities: dict[str, torch.Tensor] = {}
        self._snap_batches: dict[str, int] = defaultdict(int)
        self.max_abs: dict[str, float] = {}
        self._handles = []
        self._frozen_weight_cache: OrderedDict[
            tuple[int, str], torch.Tensor
        ] | None = None
        self._frozen_weight_format_cache: OrderedDict[
            tuple[str, str, int, str, str], torch.Tensor
        ] = (
            _SHARED_FROZEN_WEIGHT_FORMAT_CACHE
            if _env_truthy("PRISMAQUANT_SHARED_WEIGHT_FORMAT_CACHE")
            else OrderedDict()
        )
        self._fused_forward_originals: list[tuple[nn.Module, object]] = []
        self._fused_nvfp4_weight_cache: OrderedDict[
            tuple[str, str, str, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = OrderedDict()
        self._materialized_frozen_weight_depth = 0
        self._frozen_weight_cache_evictions = 0
        self._frozen_weight_cache_eviction_reported = False
        register_budget_evictor(self)

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    def install(self) -> None:
        for plan in self.plans:
            if self._try_install_nvfp4_fused_forward(plan):
                continue
            self._install_packed_expert_activation_quant(plan)
            self._handles.append(
                plan.module.register_forward_pre_hook(
                    self._make_pre_hook(plan),
                    with_kwargs=True,
                )
            )
            self._handles.append(
                plan.module.register_forward_hook(
                    self._make_post_hook(plan),
                    with_kwargs=True,
                )
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module, original_forward in reversed(self._fused_forward_originals):
            module.forward = original_forward
        self._fused_forward_originals.clear()
        for plan in self.plans:
            self._restore_plan(plan)

    def _find_param_plan(self, name: str) -> tuple[_ModulePlan, _ParamPlan]:
        for plan in self.plans:
            for param_plan in plan.params:
                if param_plan.name == name:
                    return plan, param_plan
        raise KeyError(f"no quantized parameter named {name!r}")

    def _quantized_weight_for(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
        spec: fr.FormatSpec,
    ) -> torch.Tensor | None:
        param = getattr(plan.module, param_plan.attr)
        if not isinstance(param, torch.nn.Parameter) or param.is_meta:
            return None
        fmt = fr.canonical_format_name(spec.name)
        # Include production-cache identity in the key so a SHARED
        # frozen_weight_format_cache that's seen multiple instances
        # (with/without production cache, or different production
        # caches) doesn't return stale entries.  Same-instance reuse
        # across polish trials still hits because id() is stable.
        prod_id = (
            id(self._production_weight_cache)
            if self._production_weight_cache is not None else 0
        )
        cache_key = (
            param_plan.name,
            fmt,
            int(param.data_ptr()),
            str(param.device),
            str(param.dtype),
            prod_id,
        )
        q = self._frozen_weight_format_cache.get(cache_key)
        if q is None:
            enforce_gpu_memory_budget(
                [self],
                device=param.device if param.device.type == "cuda" else None,
                reason="frozen weight cache fill",
            )
            production = (
                self._production_weight_cache.get(param_plan.name, fmt)
                if self._production_weight_cache is not None
                else None
            )
            if production is not None:
                q = production.to(
                    device=param.device,
                    dtype=param.dtype,
                ).contiguous()
            else:
                from .nvfp4_cb_footprint import is_cb_format

                if is_cb_format(fmt):
                    raise RuntimeError(
                        f"production_weight_cache is required for CB fallback "
                        f"({param_plan.name!r}, {fmt!r}); the registry path is "
                        "unweighted legacy rendering and cannot represent the "
                        "stamped production serialization contract"
                    )
                if (
                    self._production_weight_cache is not None
                    and fmt != "BF16"
                    and _env_truthy(
                        "PRISMAQUANT_STRICT_PRODUCTION_CACHE",
                        default=True,
                    )
                ):
                    raise RuntimeError(
                        f"production_weight_cache miss for "
                        f"({param_plan.name!r}, {fmt!r}); set "
                        f"PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 to fall back "
                        f"to RTN, or rebuild the cache to cover this Linear."
                    )
                original = param.data.detach().clone()
                q = spec.quantize_dequantize(original).to(
                    device=param.device,
                    dtype=param.dtype,
                ).contiguous()
            # cache_key now includes production-cache identity, so we
            # can safely populate the shared cache regardless of
            # production-active state.  Different production caches
            # (or no cache) get distinct keys; no cross-contamination.
            if self._frozen_weight_cache_max_entries() > 0:
                self._frozen_weight_format_cache[cache_key] = q
                self._evict_frozen_weight_format_cache_to_limit()
            enforce_gpu_memory_budget(
                [self],
                device=param.device if param.device.type == "cuda" else None,
                reason="frozen weight cache fill",
            )
        else:
            self._frozen_weight_format_cache.move_to_end(cache_key)
        return q

    def build_frozen_weight_cache(self) -> dict[tuple[int, str], torch.Tensor]:
        cache: OrderedDict[tuple[int, str], torch.Tensor] = OrderedDict()
        for plan in self.plans:
            seen_attrs: set[str] = set()
            for param_plan in plan.params:
                if param_plan.attr in seen_attrs:
                    continue
                seen_attrs.add(param_plan.attr)
                q = self._quantized_weight_for(plan, param_plan, param_plan.spec)
                if q is None:
                    continue
                cache[(id(plan.module), param_plan.attr)] = q
        self._frozen_weight_cache = cache
        return cache

    @contextmanager
    def frozen_weight_cache(self) -> Iterator["PerturbedActivationCache"]:
        previous = self._frozen_weight_cache
        self.build_frozen_weight_cache()
        try:
            yield self
        finally:
            self._frozen_weight_cache = previous
            self._emit_frozen_weight_cache_evictions()

    @contextmanager
    def materialized_frozen_weights(self) -> Iterator["PerturbedActivationCache"]:
        """Apply the active frozen weights to modules for whole-forward reuse."""
        if self._frozen_weight_cache is None:
            raise RuntimeError("frozen weight cache is not active")
        if self._materialized_frozen_weight_depth > 0:
            self._materialized_frozen_weight_depth += 1
            try:
                yield self
            finally:
                self._materialized_frozen_weight_depth -= 1
            return

        originals: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        seen_keys: set[tuple[int, str]] = set()
        self._materialized_frozen_weight_depth = 1
        try:
            for plan in self.plans:
                for param_plan in plan.params:
                    cache_key = (id(plan.module), param_plan.attr)
                    if cache_key in seen_keys:
                        continue
                    seen_keys.add(cache_key)
                    param = getattr(plan.module, param_plan.attr)
                    if not isinstance(param, torch.nn.Parameter) or param.is_meta:
                        continue
                    q = self._frozen_weight_cache.get(cache_key)
                    if q is None:
                        continue
                    self._frozen_weight_cache.move_to_end(cache_key)
                    originals.append((param, param.data.detach().clone()))
                    param.data.copy_(q.to(device=param.device, dtype=param.dtype))
            yield self
        finally:
            for param, original in reversed(originals):
                param.data.copy_(original.to(device=param.device, dtype=param.dtype))
            self._materialized_frozen_weight_depth = 0

    def set_frozen_weight_format(self, name: str, fmt: str) -> None:
        if self._frozen_weight_cache is None:
            raise RuntimeError("frozen weight cache is not active")
        plan, param_plan = self._find_param_plan(name)
        spec = fr.get_format(fmt)
        q = self._quantized_weight_for(plan, param_plan, spec)
        if q is None:
            return
        self._frozen_weight_cache[(id(plan.module), param_plan.attr)] = q
        self._frozen_weight_cache.move_to_end((id(plan.module), param_plan.attr))
        param_plan.spec = spec

    @contextmanager
    def temporary_frozen_weight_format(
        self,
        name: str,
        fmt: str,
    ) -> Iterator["PerturbedActivationCache"]:
        with self.override({name: fmt}):
            yield self

    @contextmanager
    def override(
        self,
        assignment_delta: Mapping[str, str],
    ) -> Iterator["PerturbedActivationCache"]:
        if self._frozen_weight_cache is None:
            raise RuntimeError("frozen weight cache is not active")
        previous: list[
            tuple[tuple[int, str], torch.Tensor | None, _ParamPlan, fr.FormatSpec]
        ] = []
        for name, fmt in assignment_delta.items():
            plan, param_plan = self._find_param_plan(name)
            cache_key = (id(plan.module), param_plan.attr)
            previous.append(
                (
                    cache_key,
                    self._frozen_weight_cache.get(cache_key),
                    param_plan,
                    param_plan.spec,
                )
            )
            self.set_frozen_weight_format(name, fmt)
        try:
            yield self
        finally:
            for cache_key, previous_q, param_plan, previous_spec in reversed(previous):
                if previous_q is None:
                    self._frozen_weight_cache.pop(cache_key, None)
                else:
                    self._frozen_weight_cache[cache_key] = previous_q
                    self._frozen_weight_cache.move_to_end(cache_key)
                param_plan.spec = previous_spec

    def _capture(self, plan: _ModulePlan, x: torch.Tensor) -> None:
        if not self.capture_inputs:
            return
        flat = x.detach().reshape(-1, x.size(-1))
        for name in plan.cache_names:
            mx = float(flat.abs().max().item())
            if mx > self.max_abs.get(name, 0.0):
                self.max_abs[name] = mx
            if self.input_rows <= 0:
                continue
            self._reservoir_update(name, flat)

    def _reservoir_update(self, name: str, flat: torch.Tensor) -> None:
        """Fold one capture batch into ``name``'s bounded uniform reservoir.

        M8 fix: the old path kept the FIRST ``input_rows`` rows of the
        calibration stream (``need = input_rows - rows_got``; all later
        batches skipped), so with default sizes the entire perturbed-X
        second moment came from calibration document #1.  This is the
        same priority-reservoir scheme as
        ``activation_sampling.update_priority_reservoir`` — uniform
        without replacement over ALL rows seen, storage bounded at
        ``input_rows`` — with two deltas that matter here:

          * priorities come from ``SharedRowSubsampler.batch_priorities``
            (keyed by fused-sibling group + batch index), so gate/up and
            q/k/v siblings keep IDENTICAL row sets across the whole
            stream, not just within one call;
          * only surviving rows are copied device→CPU
            (``update_priority_reservoir`` concatenates the full incoming
            batch onto the CPU reservoir first, which would move every
            calibration activation over the bus per module per batch).
        """
        limit = self.input_rows
        batch_index = self._snap_batches[name]
        self._snap_batches[name] = batch_index + 1
        new_pri = self.subsampler.batch_priorities(
            name, batch_index, int(flat.size(0))
        )
        cur_rows = self._snap_rows.get(name)
        cur_pri = self._snap_priorities.get(name)
        n_cur = 0 if cur_rows is None else int(cur_rows.size(0))
        merged_pri = (
            new_pri if n_cur == 0 else torch.cat([cur_pri, new_pri], dim=0)
        )
        if int(merged_pri.numel()) <= limit:
            incoming = flat.to("cpu")
            if incoming is flat:
                # `.to("cpu")` is a no-op for CPU inputs; clone so the
                # reservoir never aliases live activation storage.
                incoming = incoming.clone()
            self._snap_rows[name] = (
                incoming
                if cur_rows is None
                else torch.cat([cur_rows, incoming], dim=0)
            )
            self._snap_priorities[name] = merged_pri
            return
        keep = torch.topk(merged_pri, k=limit, largest=True, sorted=False).indices
        # Ascending order keeps retained rows in stream order and makes
        # the old-rows/new-rows concatenation below line up with the
        # reordered priorities (old indices < n_cur <= new indices).
        keep = torch.sort(keep).values
        keep_old = keep[keep < n_cur]
        keep_new = keep[keep >= n_cur] - n_cur
        parts: list[torch.Tensor] = []
        if keep_old.numel():
            parts.append(cur_rows.index_select(0, keep_old))
        if keep_new.numel():
            parts.append(
                flat.index_select(0, keep_new.to(flat.device)).to("cpu")
            )
        self._snap_rows[name] = (
            parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        )
        self._snap_priorities[name] = merged_pri.index_select(0, keep)

    def _apply_weight_quant(self, plan: _ModulePlan) -> None:
        plan.active_originals.clear()
        if self._materialized_frozen_weight_depth > 0:
            return
        if _env_truthy("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", default=False):
            # Caller (e.g. WeightSession) has installed the desired weights
            # directly on model.params; we just observe + activation-
            # quantize, no clone/restore.  Saves ~50 MB clone per module
            # on the hot path and lets polish on big models avoid the
            # cumulative-clone OOM.
            return
        seen_attrs: set[str] = set()
        for param_plan in plan.params:
            if param_plan.attr in seen_attrs:
                continue
            seen_attrs.add(param_plan.attr)
            param = getattr(plan.module, param_plan.attr)
            if not isinstance(param, torch.nn.Parameter) or param.is_meta:
                continue
            original = param.data.detach().clone()
            q = None
            if self._frozen_weight_cache is not None:
                cache_key = (id(plan.module), param_plan.attr)
                q = self._frozen_weight_cache.get(cache_key)
                if q is not None:
                    self._frozen_weight_cache.move_to_end(cache_key)
            if q is None and _env_truthy("PRISMAQUANT_SHARED_WEIGHT_FORMAT_CACHE"):
                q = self._quantized_weight_for(plan, param_plan, param_plan.spec)
            if q is None and self._production_weight_cache is not None:
                fmt_canon = fr.canonical_format_name(param_plan.spec.name)
                production = self._production_weight_cache.get(
                    param_plan.name, fmt_canon,
                )
                if production is not None:
                    q = production.to(
                        device=param.device, dtype=param.dtype,
                    ).contiguous()
                elif (
                    fmt_canon != "BF16"
                    and _env_truthy(
                        "PRISMAQUANT_STRICT_PRODUCTION_CACHE",
                        default=True,
                    )
                ):
                    raise RuntimeError(
                        f"production_weight_cache miss for "
                        f"({param_plan.name!r}, {fmt_canon!r}); set "
                        f"PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 to allow "
                        f"RTN fallback."
                    )
            if q is None:
                from .nvfp4_cb_footprint import is_cb_format

                fmt_canon = fr.canonical_format_name(param_plan.spec.name)
                if is_cb_format(fmt_canon):
                    raise RuntimeError(
                        f"production_weight_cache is required for CB fallback "
                        f"({param_plan.name!r}, {fmt_canon!r}); refusing an "
                        "unweighted legacy registry render"
                    )
                if fr.is_tessera_format_name(fmt_canon):
                    raise RuntimeError(
                        f"production_weight_cache is required for Tessera "
                        f"({param_plan.name!r}, {fmt_canon!r}); the registry "
                        "render is a weights-only reconstruction, not the "
                        "decoded wire and not the H-aware encode that ships"
                    )
                q = param_plan.spec.quantize_dequantize(original)
            if q is None:
                continue
            param.data.copy_(q.to(device=param.device, dtype=param.dtype))
            plan.active_originals.append((param, original))

    def _active_activation_spec(self, plan: _ModulePlan) -> fr.FormatSpec | None:
        if not self.include_activation_quant:
            return None
        low_act = {
            p.spec.name: p.spec
            for p in plan.params
            if p.spec.act_quant_changes_input
        }
        if len(low_act) == 1:
            return next(iter(low_act.values()))
        return None

    def _served_measurement_units(self) -> list[tuple[str, object]]:
        """``(name, contract)`` for every member the hook prices as served.

        One enumeration for both preflights below, so "which units does this
        cache measure as served, and under whose contract" cannot be answered
        two ways.  The contract travels with the name because the G rule is
        the SPEC's, never a name comparison (#205).
        """
        if not self.include_activation_quant:
            return []
        units: list[tuple[str, object]] = []
        for plan in self.plans:
            members = self._packed_act_plan(plan)
            if members is None:
                act_spec = self._active_activation_spec(plan)
                contract = getattr(act_spec, "static_activation_contract", None)
                if contract is None or not contract.measured_as_served:
                    continue
                # The name the hook looks the scale up under: the dense
                # ``weight`` member (``_module_input_member_name``), else
                # every member -- without an input tensor the structural
                # tie-break cannot run, and refusing one name too many is
                # the safe side.
                weight = next(
                    (p.name for p in plan.params if p.attr == "weight"), None)
                names = [weight] if weight is not None else plan.cache_names
                units.extend((name, contract) for name in names)
            else:
                units.extend(
                    (m.name, m.spec.static_activation_contract)
                    for m in members
                    if getattr(m.spec.static_activation_contract,
                               "measured_as_served", False)
                )
        return units

    def served_activation_scale_gaps(self) -> list[str]:
        """Names this cache would have to REFUSE in the hook, listed up front.

        A member whose spec is measured under the served static-scale
        contract (``FormatSpec.static_activation_contract.measured_as_served``,
        a Tessera W4A4 rung) needs its calibrated maximum in this cache's
        scale identity (``activation_max_abs`` from the production cache);
        ``_activation_qdq`` refuses it by name otherwise.  Consumers that
        measure (``kl_measurement.measure_assignment_kl``) ask this before the
        first forward so the refusal names every unit at once instead of the
        first hook the model happens to reach.  Capture-only builders, which
        run before any maximum exists, are not asked.
        """
        gaps: set[str] = set()
        for name, _contract in self._served_measurement_units():
            value = _activation_max_abs_lookup(self._activation_scales, name)
            if value is None or float(value) <= 0.0:
                gaps.add(name)
        return sorted(gaps)

    def served_activation_policy_conflicts(self) -> list[str]:
        """Names whose cached cost was priced at a different static G (#227).

        The sibling of :meth:`served_activation_scale_gaps`: that one asks
        whether a unit HAS a calibrated maximum, which stays true across a
        change of input-global-scale policy; this one asks whether the G that
        maximum now derives is the G the unit's retained render score was
        priced at.  Asked before the first forward for the same reason -- the
        refusal names every affected unit rather than the first one the model
        reaches -- and answered from the cache's own score provenance, never
        from the environment.
        """
        conflicts: set[str] = set()
        if not self._priced_input_global_scales:
            return []
        for name, contract in self._served_measurement_units():
            priced = _activation_max_abs_lookup(
                self._priced_input_global_scales, name)
            max_abs = _activation_max_abs_lookup(self._activation_scales, name)
            if priced is None or max_abs is None or float(max_abs) <= 0.0:
                continue
            applied = contract.input_global_scale_from_max_abs(float(max_abs))
            if float(priced) != float(applied):
                conflicts.add(name)
        return sorted(conflicts)

    def _nvfp4_fused_param_plan(self, plan: _ModulePlan) -> _ParamPlan | None:
        if not _env_truthy("PRISMAQUANT_FUSED_KERNEL_NVFP4"):
            return None
        # When a production cache is active, the fused fast path's
        # `nvfp4_pack_weight` re-computes per-group scales locally and
        # ignores the cache's joint NVFP4 sibling globals — so the
        # packed FP4 codes diverge from what the export would produce.
        # Refuse to use the fast path in that mode unless the user
        # explicitly opts in via PRISMAQUANT_FUSED_KERNEL_OVER_PROD_CACHE.
        if (
            self._production_weight_cache is not None
            and not _env_truthy("PRISMAQUANT_FUSED_KERNEL_OVER_PROD_CACHE")
        ):
            return None
        if not isinstance(plan.module, nn.Linear) or len(plan.params) != 1:
            return None
        param_plan = plan.params[0]
        if param_plan.attr != "weight":
            return None
        if fr.canonical_format_name(param_plan.spec.name) != "NVFP4":
            return None
        act_spec = self._active_activation_spec(plan)
        if act_spec is None or fr.canonical_format_name(act_spec.name) != "NVFP4":
            return None
        return param_plan

    def _try_install_nvfp4_fused_forward(self, plan: _ModulePlan) -> bool:
        param_plan = self._nvfp4_fused_param_plan(plan)
        if param_plan is None:
            return False
        try:
            from prismaquant.kernels.nvfp4_fused import nvfp4_fused_aw_matmul  # noqa: F401
        except Exception:
            return False

        module = plan.module
        original_forward = module.forward

        def _forward(x, *args, **kwargs):
            if args or kwargs or not isinstance(x, torch.Tensor):
                return original_forward(x, *args, **kwargs)
            return self._nvfp4_fused_linear_forward(plan, param_plan, x)

        module.forward = _forward
        self._fused_forward_originals.append((module, original_forward))
        return True

    def _weight_for_reference_forward(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
    ) -> torch.Tensor:
        param = getattr(plan.module, param_plan.attr)
        if not isinstance(param, torch.nn.Parameter) or param.is_meta:
            return param
        q = None
        if self._frozen_weight_cache is not None:
            cache_key = (id(plan.module), param_plan.attr)
            q = self._frozen_weight_cache.get(cache_key)
            if q is not None:
                self._frozen_weight_cache.move_to_end(cache_key)
        if q is None:
            q = self._quantized_weight_for(plan, param_plan, param_plan.spec)
        if q is None:
            return param
        return q.to(device=param.device, dtype=param.dtype)

    def _reference_linear_forward(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
        x: torch.Tensor,
    ) -> torch.Tensor:
        act_spec = self._active_activation_spec(plan)
        if act_spec is not None:
            # MED-3: act-clip the input to the calibrated max_abs before
            # per-group RTN.  The dynamic per-group quantizer in
            # `act_spec.activation_quantize_dequantize` would otherwise
            # set its scales from the input's per-group max — outliers
            # then dominate.  Production export does the same clipping
            # as `_resolve_act_clip_quantile`, so this matches what the
            # shipped artifact sees at runtime.  `Q(x/s)*s == Q(x)` for
            # purely dynamic Q, so the previous "pre-scale + post-multiply"
            # formulation was a no-op (codex round-3).
            x = _activation_qdq(
                x, act_spec, self._activation_scales, param_plan.name,
                self._priced_input_global_scales,
            )
        weight = self._weight_for_reference_forward(plan, param_plan)
        return F.linear(x, weight, plan.module.bias)

    def _packed_nvfp4_weight_for(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        param = getattr(plan.module, param_plan.attr)
        if not isinstance(param, torch.nn.Parameter) or param.is_meta:
            raise RuntimeError("cannot pack a missing or meta Linear weight")
        cache_key = (
            param_plan.name,
            str(param.device),
            str(param.dtype),
            int(param.data_ptr()),
        )
        packed = self._fused_nvfp4_weight_cache.get(cache_key)
        if packed is None:
            from prismaquant.kernels.nvfp4_fused import nvfp4_pack_weight

            enforce_gpu_memory_budget(
                [self],
                device=param.device if param.device.type == "cuda" else None,
                reason="NVFP4 packed weight cache fill",
            )
            # HIGH: prefer the production cache's GPTQ + scale_sweep
            # weight when present.  Without this, the fused NVFP4 fast
            # path packs the raw BF16 param and bypasses the entire
            # production cache (silently runs RTN-equivalent weights
            # through the kernel).  Strict mode raises on miss so the
            # fast path matches the slow-path miss semantics.
            source = param.detach()
            if self._production_weight_cache is not None:
                w_dq = self._production_weight_cache.get(
                    param_plan.name, "NVFP4",
                )
                if w_dq is not None:
                    source = w_dq.to(
                        device=param.device, dtype=param.dtype,
                    ).contiguous()
                elif _env_truthy(
                    "PRISMAQUANT_STRICT_PRODUCTION_CACHE",
                    default=True,
                ):
                    raise RuntimeError(
                        f"production_weight_cache miss for "
                        f"({param_plan.name!r}, 'NVFP4') on the fused "
                        f"NVFP4 fast path; set "
                        f"PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 to allow "
                        f"raw-weight fallback or rebuild the cache."
                    )
            packed = nvfp4_pack_weight(source)
            self._fused_nvfp4_weight_cache[cache_key] = packed
            self._fused_nvfp4_weight_cache.move_to_end(cache_key)
            enforce_gpu_memory_budget(
                [self],
                device=param.device if param.device.type == "cuda" else None,
                reason="NVFP4 packed weight cache fill",
            )
        else:
            self._fused_nvfp4_weight_cache.move_to_end(cache_key)
        return packed

    def _frozen_weight_cache_max_entries(self) -> int:
        return env_int("PRISMAQUANT_FROZEN_WEIGHT_CACHE_MAX_ENTRIES", 400)

    def _evict_frozen_weight_format_cache_to_limit(self) -> None:
        max_entries = self._frozen_weight_cache_max_entries()
        if max_entries <= 0:
            evicted = len(self._frozen_weight_format_cache)
            self._frozen_weight_format_cache.clear()
            self._frozen_weight_cache_evictions += evicted
            return
        while len(self._frozen_weight_format_cache) > max_entries:
            self._frozen_weight_format_cache.popitem(last=False)
            self._frozen_weight_cache_evictions += 1

    def evict_oldest_for_memory_budget(self) -> bool:
        if self._frozen_weight_format_cache:
            self._frozen_weight_format_cache.popitem(last=False)
            self._frozen_weight_cache_evictions += 1
            return True
        if self._fused_nvfp4_weight_cache:
            self._fused_nvfp4_weight_cache.popitem(last=False)
            return True
        if self._frozen_weight_cache:
            self._frozen_weight_cache.popitem(last=False)
            self._frozen_weight_cache_evictions += 1
            return True
        return False

    def _emit_frozen_weight_cache_evictions(self) -> None:
        if (
            self._frozen_weight_cache_evictions <= 0
            or self._frozen_weight_cache_eviction_reported
        ):
            return
        self._frozen_weight_cache_eviction_reported = True
        print(
            "[frozen-weight-cache] evicted "
            f"{self._frozen_weight_cache_evictions} entries "
            f"(max_entries={self._frozen_weight_cache_max_entries()})",
            file=sys.stderr,
            flush=True,
        )

    def _nvfp4_fused_linear_forward(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
        x: torch.Tensor,
    ) -> torch.Tensor:
        self._capture(plan, x)
        x_runtime = x
        act_spec = self._active_activation_spec(plan)
        fused_active = (
            fr.canonical_format_name(param_plan.spec.name) == "NVFP4"
            and act_spec is not None
            and fr.canonical_format_name(act_spec.name) == "NVFP4"
            and x_runtime.is_cuda
            and x_runtime.shape[-1] % 16 == 0
        )
        if not fused_active:
            return self._reference_linear_forward(plan, param_plan, x)

        from prismaquant.kernels.nvfp4_fused import nvfp4_fused_aw_matmul

        w_packed, w_scales, w_global_scale = self._packed_nvfp4_weight_for(
            plan, param_plan
        )
        flat_x = x_runtime.reshape(-1, x_runtime.shape[-1])
        # MED-3: act-clip the activation to the calibrated max_abs before
        # the fused kernel's internal per-group RTN.  Same rationale as
        # ``_reference_linear_forward``: pre-scale + post-multiply
        # cancels under dynamic per-group RTN, but clipping forces
        # outliers to the calibrated range so per-group scales are
        # bounded — matching production's act-clip semantics.
        flat_x = _maybe_clip_activations(
            flat_x, self._activation_scales, param_plan.name,
        )
        out = nvfp4_fused_aw_matmul(flat_x, w_packed, w_scales, w_global_scale)
        out = out.reshape(*x_runtime.shape[:-1], plan.module.out_features)
        if plan.module.bias is not None:
            out = out + plan.module.bias.to(device=out.device, dtype=out.dtype)
        return out

    def _restore_plan(self, plan: _ModulePlan) -> None:
        if _env_truthy("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", default=False):
            # Mirror of _apply_weight_quant's bypass — WeightSession
            # owns weight transitions, so there's nothing to restore.
            return
        for param, original in reversed(plan.active_originals):
            param.data.copy_(original.to(device=param.device, dtype=param.dtype))
        plan.active_originals.clear()

    def _packed_act_plan(self, plan: _ModulePlan) -> list[_ParamPlan] | None:
        """The per-projection params of a packed-experts plan, or None.

        A packed-experts module owns several 3-D projection parameters and is
        not an ``nn.Linear``, so the module-level pre-hook can only ever see
        ONE of their inputs -- the module input, which is gate_up's. down_proj
        consumes the post-SwiGLU intermediate produced INSIDE the forward, and
        no hook on the module boundary can reach it.

        That is not a cosmetic gap. vLLM's ``CompressedTensorsW4A4Nvfp4MoEMethod``
        registers BOTH ``w13_input_global_scale`` and ``w2_input_global_scale``:
        the served runtime quantizes both activations. A gate that emulates only
        one of them measures a cheaper model than the one that ships, on the
        half of the MoE FLOPs it left alone -- and it is the selecting gate, so
        the error goes straight into which assignment is chosen.
        """
        if not self.include_activation_quant or len(plan.params) < 2:
            return None
        if isinstance(plan.module, nn.Linear):
            return None
        members = [
            p for p in plan.params
            if getattr(p.spec, "act_quant_changes_input", False)
            and getattr(getattr(plan.module, p.attr, None), "ndim", 0) == 3
        ]
        return members or None

    def _install_packed_expert_activation_quant(self, plan: _ModulePlan) -> None:
        """Quantize each expert-slice ``F.linear`` input with ITS OWN spec.

        Same interception the probe uses to capture packed-expert Fisher
        (``sensitivity_probe.install_packed_expert_hooks``): swap ``F.linear``
        for the duration of the experts-module forward and dispatch on whether
        the weight is a dim-0 slice of one of this plan's packed parameters.
        Eval-time only -- no autograd Function, no gradient path.

        Each projection uses its own calibrated activation scale, keyed by its
        own param name. The module-level pre-hook's act-qdq is suppressed for
        these plans (see ``_make_pre_hook``) so gate_up's input is quantized
        exactly once, here, rather than once there and once again inside.
        """
        members = self._packed_act_plan(plan)
        if members is None:
            return
        from prismaquant.sensitivity_probe import _packed_expert_slice_index

        module = plan.module
        original_forward = module.forward
        owner = self

        def _forward(*args, **kwargs):
            targets: dict[int, _ParamPlan] = {}
            params: dict[int, torch.Tensor] = {}
            for member in members:
                param = getattr(module, member.attr, None)
                if not isinstance(param, torch.Tensor) or param.ndim != 3:
                    continue
                targets[id(param)] = member
                params[id(param)] = param
            if not targets:
                return original_forward(*args, **kwargs)
            orig_linear = F.linear

            def _intercepting_linear(input, weight, bias=None):
                base = weight._base if weight._is_view() else weight
                member = targets.get(id(base))
                if member is not None and isinstance(input, torch.Tensor):
                    if _packed_expert_slice_index(
                            weight, params[id(base)]) is not None:
                        input = _activation_qdq(
                            input, member.spec, owner._activation_scales,
                            member.name,
                            owner._priced_input_global_scales)
                return orig_linear(input, weight, bias)

            F.linear = _intercepting_linear
            try:
                return original_forward(*args, **kwargs)
            finally:
                F.linear = orig_linear

        module.forward = _forward
        self._fused_forward_originals.append((module, original_forward))

    def _make_pre_hook(self, plan: _ModulePlan):
        def _pre_hook(_module, args, kwargs):
            where, key, x = _first_tensor_location(args, kwargs)
            if isinstance(x, torch.Tensor):
                self._capture(plan, x)
                member_name = _module_input_member_name(plan, x)
                x_runtime = x
                act_spec = self._active_activation_spec(plan)
                if self._packed_act_plan(plan) is not None:
                    # Handled per projection inside the forward, where
                    # down_proj's input is reachable and each projection gets
                    # its own calibrated scale. Quantizing here too would put
                    # gate_up's input through the quantizer twice.
                    act_spec = None
                if act_spec is not None:
                    # MED-3: act-clip to the calibrated max_abs before the
                    # quantizer, so outliers don't dominate per-group
                    # scales.  See ``_maybe_clip_activations`` for the
                    # math; pre-scale + post-multiply was a no-op (codex
                    # round-3 caught Q(x/s)*s == Q(x)).
                    x_runtime = _activation_qdq(
                        x_runtime, act_spec, self._activation_scales,
                        member_name, self._priced_input_global_scales,
                    )
                if x_runtime is not x:
                    args, kwargs = _replace_tensor_input(
                        args, kwargs, where, key, x_runtime,
                    )
            self._apply_weight_quant(plan)
            return args, kwargs

        return _pre_hook

    def _make_post_hook(self, plan: _ModulePlan):
        def _post_hook(_module, _args, _kwargs, output):
            self._restore_plan(plan)
            return output

        return _post_hook

    def finalize(self) -> dict:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for name, rows in self._snap_rows.items():
            if rows is None or rows.size(0) == 0:
                continue
            x = rows[:self.input_rows].to(torch.bfloat16).contiguous()
            write_activation_cache_entry(self.cache_dir, name, x)
            written.append(name)
        return {
            "cache_dir": str(self.cache_dir),
            "written": sorted(written),
            "missing": sorted(self.missing),
            "skipped_activation_quant": self.skipped,
        }


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def iter_calibration_forwards(
    calibration_data,
    device: torch.device,
    *,
    microbatch_size: int = 1,
):
    """Yield (args, kwargs) for one forward pass per calibration microbatch.

    ``microbatch_size`` controls how many calibration rows are stacked into
    each forward — default 1 preserves the historical one-sample-at-a-time
    behaviour every existing caller relies on.  Callers that want to amortize
    Python and kernel launch overhead can request a larger microbatch; the
    yielded batch dim becomes ``min(microbatch_size, remaining_rows)``.
    """
    if isinstance(calibration_data, torch.Tensor):
        n = int(calibration_data.size(0))
        m = max(1, int(microbatch_size))
        for i in range(0, n, m):
            yield (calibration_data[i:i + m].to(device),), {}
        return
    if isinstance(calibration_data, Mapping):
        yield (), {k: _to_device(v, device) for k, v in calibration_data.items()}
        return
    for sample in calibration_data:
        if isinstance(sample, torch.Tensor):
            yield (sample.to(device),), {}
        elif isinstance(sample, Mapping):
            yield (), {k: _to_device(v, device) for k, v in sample.items()}
        elif isinstance(sample, tuple):
            yield tuple(_to_device(v, device) for v in sample), {}
        else:
            yield (sample,), {}


@torch.no_grad()
def capture_perturbed_activation_cache(
    model: nn.Module,
    assignment: Mapping[str, str],
    calibration_data,
    cache_dir: str | Path,
    *,
    input_rows: int = 256,
    profile=None,
    cal_hash: str | None = None,
) -> dict:
    """Run calibration forwards and write an ActivationIndex-compatible cache."""
    cal_hash = cal_hash or calibration_data_hash(calibration_data)
    builder = PerturbedActivationCache(
        model,
        assignment,
        cache_dir,
        input_rows=input_rows,
        cal_hash=cal_hash,
        profile=profile,
    )
    device = _model_device(model)
    builder.install()
    try:
        # PRISMAQUANT_L2_CUDA_GRAPHS is intentionally not applied here.
        # These forwards must execute Python hooks on every batch to snapshot
        # perturbed-X activations; CUDA graph replay would skip those hooks and
        # silently under-fill the activation cache.
        for args, kwargs in iter_calibration_forwards(calibration_data, device):
            model(*args, **kwargs)
    finally:
        builder.remove()
    manifest = builder.finalize()
    manifest["calibration_hash"] = cal_hash
    manifest["input_rows"] = int(input_rows)
    with open(Path(cache_dir) / "perturbed_x_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def stage_text_only_under_work_root(model_path: str, work_root: str | Path) -> str:
    """Text-only staging equivalent to sensitivity_probe, but never under /tmp.

    Thin wrapper around `sensitivity_probe._stage_text_only_impl` (issue
    #210: one home for the default strip-key list and the staging steps,
    shared with `sensitivity_probe.stage_text_only`). This name and
    signature stay so no caller moves; only the staging root differs
    (an explicit, caller-owned `work_root`, never /tmp, with no `atexit`
    registration).
    """
    from .sensitivity_probe import _stage_text_only_impl
    return _stage_text_only_impl(model_path, staging_root=work_root)


def load_text_model_under_work_root(
    model_path: str,
    *,
    device: str,
    dtype: torch.dtype,
    work_root: str | Path,
    device_map: str | None = None,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    staged = stage_text_only_under_work_root(model_path, work_root)
    load_device_map = device_map if device_map is not None else device
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": load_device_map,
        "low_cpu_mem_usage": False,
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
    except ValueError as exc:
        if "requires `accelerate`" not in str(exc) and "requires accelerate" not in str(exc):
            raise
        load_kwargs.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        model.to(torch.device(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
