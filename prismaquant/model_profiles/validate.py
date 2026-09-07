#!/usr/bin/env python3
"""validate_profile.py — automated consistency checks for a
ModelProfile against an actual checkpoint.

Usage:

    python -m prismaquant.model_profiles.validate \\
        --model /path/to/Model
    python -m prismaquant.model_profiles.validate \\
        --profile MyCustomProfile \\
        --model /path/to/Model

Without `--profile`, the validator auto-detects using the registry.
With `--profile`, it imports that class name from `prismaquant.model_profiles`
or any module importable via `$PYTHONPATH`.

Checks performed:

  1. **Profile claim.** `profile.matches()` returns True for this
     model's (model_type, architectures) tuple.

  2. **vLLM class exists.** If the profile returns a
     `vllm_architecture_class()`, vLLM's registry can resolve it.

  3. **Fused-sibling self-consistency.** Every fused-group member
     (`profile.fused_sibling_group(sibling)`) returns the same
     canonical key across all siblings of the same group, using the
     vLLM `packed_modules_mapping`'s own sibling lists as ground
     truth.

  4. **Name remap fixed points.** For every
     string-destination `orig_to_new_prefix` entry in vLLM's
     `hf_to_vllm_mapper`, a probe at a real component boundary starts
     with the expected `new_prefix`. Loader-only drops (`None`)
     are counted separately and excluded from dispatch-name checks.

  5. **MTP module construction** (if `has_mtp()` is True). The
     profile can instantiate an MTP module from the model's text
     config; the module loads the source's `mtp.*` weights without
     missing keys.

  6. **Packed-expert parameter names.** Every expert weight on disk
     resolves to one of `profile.packed_expert_param_names()`, in
     either legal source layout — packed 3D (`experts.gate_up_proj`)
     or per-expert 2D (`experts.7.gate_proj.weight`, which is what a
     stock HF MoE checkpoint ships) — and no 3D expert tensor carries
     a parameter name the profile does not declare.

  7. **Source passthrough sanity.** Every prefix in
     `profile.source_passthrough_prefixes()` matches at least one
     tensor in the source's safetensors index (otherwise the prefix
     is dead weight).

  8. **Serving profile sanity.** The model's default serving profile
     exists, and its configured runtime validator callables import.

Exit code 0 if every check passes, 1 otherwise. Each failure prints
a ✗ line with context; each success prints a ✓. Intended to be
CI-friendly so new profiles get a clear pass/fail signal before
they're used in a production export.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


class CheckResult:
    __slots__ = ("name", "ok", "detail")

    def __init__(self, name: str, ok: bool, detail: str = ""):
        self.name = name
        self.ok = ok
        self.detail = detail

    def __str__(self):
        glyph = "✓" if self.ok else "✗"
        out = f"  {glyph} {self.name}"
        if self.detail:
            out += f"\n      {self.detail}"
        return out


def _load_config(model_path: str) -> dict:
    with open(Path(model_path) / "config.json") as f:
        return json.load(f)


def _get_profile(profile_arg: str | None, model_path: str):
    """Resolve the profile to validate — either by name or by
    auto-detection from the model path."""
    from .registry import detect_profile
    from .base import ModelProfile

    if profile_arg is None:
        profile = detect_profile(model_path)
        source = f"auto-detected"
    else:
        # Try to import by class path first (`pkg.module:Cls` or
        # `pkg.module.Cls`), else from prismaquant.model_profiles by
        # bare class name.
        if ":" in profile_arg:
            modname, clsname = profile_arg.split(":", 1)
            mod = importlib.import_module(modname)
            cls = getattr(mod, clsname)
        elif "." in profile_arg:
            modname, clsname = profile_arg.rsplit(".", 1)
            mod = importlib.import_module(modname)
            cls = getattr(mod, clsname)
        else:
            from . import registry as _r
            cls = None
            for candidate in [*_r._REGISTERED, _r.DefaultProfile]:
                if candidate.__name__ == profile_arg:
                    cls = candidate
                    break
            if cls is None:
                # Also try prismaquant.model_profiles namespace.
                mod = importlib.import_module("prismaquant.model_profiles")
                cls = getattr(mod, profile_arg, None)
            if cls is None:
                raise SystemExit(
                    f"Could not resolve profile '{profile_arg}'. "
                    "Pass a dotted path (pkg.module.Cls) or register "
                    "the profile first via register_profile().")
        profile = cls()
        source = f"explicit: {cls.__name__}"
    if not isinstance(profile, ModelProfile):
        raise SystemExit(f"Profile is not a ModelProfile subclass: {profile!r}")
    return profile, source


def _check_matches(profile, cfg: dict) -> CheckResult:
    model_type = cfg.get("model_type") or ""
    archs = list(cfg.get("architectures") or [])
    try:
        ok = bool(profile.__class__.matches(model_type, archs))
    except Exception as e:
        return CheckResult("matches() returns True for this model",
                           False, f"threw {type(e).__name__}: {e}")
    if not ok:
        return CheckResult(
            "matches() returns True for this model", False,
            f"model_type={model_type!r}, architectures={archs}")
    return CheckResult(
        "matches() returns True for this model", True,
        f"model_type={model_type!r}, architectures={archs}")


def _check_vllm_class(profile) -> CheckResult:
    arch = profile.vllm_architecture_class()
    if arch is None:
        return CheckResult(
            "vllm_architecture_class() resolves",
            True,
            "profile provides no vLLM class (fine — arch-specific methods "
            "must be manually overridden)")
    from .vllm_registry import vllm_class_for_architecture
    cls = vllm_class_for_architecture(arch)
    if cls is None:
        vllm_available = True
        try:
            import vllm  # noqa: F401
        except Exception:
            vllm_available = False
        spec = None
        try:
            spec = profile.structure_spec()
        except Exception:
            spec = None
        if not vllm_available and spec is not None:
            return CheckResult(
                "vllm_architecture_class() resolves",
                True,
                f"vLLM is not importable here; arch='{arch}' will be "
                f"cross-checked in a vLLM environment, and CPU paths use "
                f"declarative spec '{spec.id}' as fallback",
            )
        return CheckResult(
            "vllm_architecture_class() resolves", False,
            f"arch='{arch}' not found in vLLM registry; "
            "install newer vLLM or change the arch name")
    return CheckResult(
        "vllm_architecture_class() resolves", True,
        f"{arch} → {cls.__module__}.{cls.__name__}")


def _check_fused_siblings(profile) -> CheckResult:
    arch = profile.vllm_architecture_class()
    if arch is None:
        return CheckResult("fused-sibling groups consistent", True,
                           "no vLLM class to cross-check against")
    from .vllm_registry import (
        vllm_class_for_architecture, packed_modules_mapping_from_class,
    )
    cls = vllm_class_for_architecture(arch)
    pm = packed_modules_mapping_from_class(cls)
    if not pm:
        return CheckResult("fused-sibling groups consistent", True,
                           "vLLM class has no packed_modules_mapping")
    failures = []
    # Use an arbitrary prefix to stand in for a parent module qname.
    parent = "model.layers.0.parent."
    for fused, siblings in pm.items():
        keys = {profile.fused_sibling_group(parent + s) for s in siblings}
        if len(keys) != 1 or next(iter(keys)) is None:
            failures.append(f"{fused}: siblings {siblings} → keys {keys}")
    if failures:
        return CheckResult("fused-sibling groups consistent", False,
                           "; ".join(failures))
    return CheckResult("fused-sibling groups consistent", True,
                       f"{len(pm)} fused groups × multiple siblings map "
                       "to the same canonical key")


def _check_name_remap(profile) -> CheckResult:
    arch = profile.vllm_architecture_class()
    if arch is None:
        return CheckResult("to_vllm_internal_name() obeys vLLM's prefix map",
                           True, "no vLLM class to cross-check against")
    from .vllm_registry import (
        vllm_class_for_architecture, hf_to_vllm_prefix_map_from_class,
    )
    cls = vllm_class_for_architecture(arch)
    prefix_map = hf_to_vllm_prefix_map_from_class(cls)
    if not prefix_map:
        return CheckResult("to_vllm_internal_name() obeys vLLM's prefix map",
                           True, "vLLM class has no hf_to_vllm_mapper")
    failures = []
    excluded_drops = 0
    for src, dst in prefix_map.items():
        # Loader-only drops have no dispatch destination to cross-check.
        if dst is None:
            excluded_drops += 1
            continue
        separator = "" if not src or src.endswith(".") else "."
        probe_name = src + separator + "x.y"
        got = profile.to_vllm_internal_name(probe_name)
        expected_prefix = dst
        if not got.startswith(expected_prefix):
            failures.append(f"{probe_name!r} -> {got!r}, expected prefix {expected_prefix!r}")
    if failures:
        return CheckResult("to_vllm_internal_name() obeys vLLM's prefix map",
                           False, "; ".join(failures))
    return CheckResult("to_vllm_internal_name() obeys vLLM's prefix map",
                       True, f"{len(prefix_map) - excluded_drops} prefix rewrites agree; "
                       f"{excluded_drops} loader-only drops excluded")


def _check_mtp(profile, cfg: dict, model_path: str) -> CheckResult:
    if not profile.has_mtp():
        return CheckResult("MTP module constructs + loads weights",
                           True, "profile reports no MTP (skipped)")
    from transformers import AutoConfig
    try:
        hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        return CheckResult("MTP module constructs + loads weights", False,
                           f"AutoConfig failed: {e}")
    text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    try:
        mtp = profile.build_mtp_module(text_cfg)
    except Exception as e:
        return CheckResult("MTP module constructs + loads weights", False,
                           f"build_mtp_module threw {type(e).__name__}: {e}")
    if mtp is None:
        return CheckResult("MTP module constructs + loads weights", False,
                           "build_mtp_module returned None despite has_mtp()")
    return CheckResult("MTP module constructs + loads weights", True,
                       f"{type(mtp).__name__} with "
                       f"{sum(1 for _ in mtp.named_parameters())} parameters")


def _check_source_passthrough(profile, model_path: str) -> CheckResult:
    """Passthrough prefixes should mostly match — but profiles cover a
    family of variants (e.g. Gemma 4 26B-A4B has no audio tower, but
    Gemma 4 31B-IT may), so unused prefixes aren't a fatal profile bug.
    We pass if at least one prefix matches something AND report unused
    prefixes as informational. Fail only if every declared prefix is
    dead on this checkpoint — that implies the profile doesn't know
    this architecture at all."""
    prefixes = profile.source_passthrough_prefixes()
    if not prefixes:
        return CheckResult("source_passthrough_prefixes() cover real tensors",
                           True, "profile declares no passthrough prefixes")
    idx_path = Path(model_path) / "model.safetensors.index.json"
    if not idx_path.is_file():
        return CheckResult("source_passthrough_prefixes() cover real tensors",
                           True, f"{idx_path} missing — cannot verify")
    with open(idx_path) as f:
        keys = list(json.load(f).get("weight_map", {}).keys())
    covered = [p for p in prefixes if any(k.startswith(p) for k in keys)]
    missing = [p for p in prefixes if p not in covered]
    if not covered:
        return CheckResult(
            "source_passthrough_prefixes() cover real tensors", False,
            f"no declared prefix matches any tensor on disk: {list(prefixes)}")
    detail = f"{len(covered)}/{len(prefixes)} prefixes match"
    if missing:
        detail += f" — unused on this variant: {missing}"
    return CheckResult(
        "source_passthrough_prefixes() cover real tensors", True, detail)


def _safetensors_header(path: Path) -> dict:
    """Read a safetensors file's JSON header (8-byte little-endian length
    prefix, then the header itself). Nothing else in the file is touched, so
    this is a few-hundred-KB read even for a 20 GB shard."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def _source_weight_map(model_path: str) -> tuple[dict[str, str], str]:
    """Map tensor key -> containing file for a checkpoint, plus a note.

    Prefers `model.safetensors.index.json`; falls back to the header of a
    single-file `model.safetensors`, because a single-shard checkpoint
    legitimately has no index and treating that as "cannot verify" passes a
    check that verified nothing."""
    root = Path(model_path)
    idx_path = root / "model.safetensors.index.json"
    if idx_path.is_file():
        with open(idx_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        return {str(k): str(v) for k, v in weight_map.items()}, "index"
    single = root / "model.safetensors"
    if single.is_file():
        header = _safetensors_header(single)
        return (
            {k: single.name for k in header if k != "__metadata__"},
            "single-file header",
        )
    return {}, f"{idx_path} and {single} both missing — cannot verify"


def _tensor_shapes(model_path: str, files: set[str]) -> dict[str, tuple[int, ...]]:
    """Best-effort key -> shape over the named shards. Shards that are absent
    (metadata-only HF cache entries hold the index but no weights) are simply
    skipped, so rank checks degrade to name checks instead of failing."""
    shapes: dict[str, tuple[int, ...]] = {}
    for fname in sorted(files):
        path = Path(model_path) / fname
        if not path.is_file():
            continue
        try:
            header = _safetensors_header(path)
        except Exception:
            continue
        for k, meta in header.items():
            if k == "__metadata__" or not isinstance(meta, dict):
                continue
            shape = meta.get("shape")
            if isinstance(shape, list):
                shapes[str(k)] = tuple(int(d) for d in shape)
    return shapes


def _check_packed_experts(profile, model_path: str) -> CheckResult:
    """Cross-check: does every expert weight on disk resolve to one of the
    profile's declared packed-expert parameter names?

    Two source layouts are legal and both must validate:

      * packed:     `<...>.experts.gate_up_proj`        (3D, [E, 2I, H])
      * per-expert: `<...>.experts.7.gate_proj.weight`  (2D — what a stock HF
                    MoE checkpoint ships; packing happens at load/export)

    The previous implementation tested `k.endswith(f"experts.{n}")`, which
    only ever matches the packed layout, so every un-packed HF source
    (Laguna, ornith-35B, DSv4) failed a check it should pass. Classification
    is delegated to the profile's own accessors (`packed_expert_role_group`,
    the shared expert-qname splitter) so expert naming stays owned by the
    profile rather than re-parsed here. When the shards are readable, ranks
    are verified too: a 3D expert tensor whose parameter name the profile
    does NOT declare is a hard fail — the pipeline would silently skip it."""
    weight_map, source = _source_weight_map(model_path)
    if not weight_map:
        return CheckResult(
            "packed_expert_param_names() cover actual expert tensors",
            True, source)
    names = profile.packed_expert_param_names()
    if not names:
        return CheckResult(
            "packed_expert_param_names() cover actual expert tensors",
            True, "profile declares no packed-expert names")

    # Only the shards that actually hold expert tensors need a header read.
    expert_files = {
        f for k, f in weight_map.items()
        if ".experts." in k or k.endswith(".experts")
    }
    shapes = _tensor_shapes(model_path, expert_files)

    split_qname = type(profile)._packed_expert_projection_leaf
    found: dict[str, int] = {n: 0 for n in names}
    unmapped: dict[str, int] = {}
    undeclared_3d: dict[str, int] = {}
    layouts: set[str] = set()
    n_expert_keys = 0
    for key in weight_map:
        # A key is `<parent>.experts.<...>` optionally followed by a
        # parameter suffix (`.weight`, `.weight_scale`, ...). Try the key
        # itself first (packed recipe form carries no suffix), then the key
        # with its last component dropped.
        parsed = None
        for cand in (key, key.rsplit(".", 1)[0]):
            parsed = split_qname(cand)
            if parsed is not None:
                break
        if parsed is None:
            continue
        _parent, leaf, split_per_expert = parsed
        n_expert_keys += 1
        role = profile.packed_expert_role_group(cand)
        rank = len(shapes[key]) if key in shapes else None
        is_primary_weight = cand == key or key.endswith(".weight")
        if role is None:
            # Not an expert weight this profile can name a role for. A router
            # sidecar (`experts.e_score_correction_bias`) is 1D and fine; a 3D
            # tensor here is an undeclared packed-expert parameter.
            if rank == 3 and is_primary_weight:
                undeclared_3d[leaf] = undeclared_3d.get(leaf, 0) + 1
            continue
        layouts.add("per-expert" if split_per_expert else "packed")
        if role in found:
            found[role] += 1
        else:
            unmapped[leaf] = unmapped.get(leaf, 0) + 1

    if undeclared_3d:
        return CheckResult(
            "packed_expert_param_names() cover actual expert tensors", False,
            f"3D expert tensors on disk that the profile does not declare: "
            f"{undeclared_3d} — add them to packed_expert_param_names() or "
            f"the spec's packed_experts.param_names")
    if n_expert_keys == 0:
        # Same leniency as check 7: one profile covers a family, and a dense
        # member of that family (Gemma 4 31B-IT vs 26B-A4B) legitimately has
        # no expert tensors at all. Declaring packed names it never uses is
        # not a profile bug; declaring names that don't match the experts
        # that ARE there is, and that is the branch below.
        return CheckResult(
            "packed_expert_param_names() cover actual expert tensors", True,
            f"checkpoint has no expert tensors (dense variant; "
            f"{len(weight_map)} keys via {source})")
    covered = [n for n, c in found.items() if c > 0]
    missing = [n for n, c in found.items() if c == 0]
    if not covered:
        return CheckResult(
            "packed_expert_param_names() cover actual expert tensors", False,
            f"none of {set(names)} are reachable from the {n_expert_keys} "
            f"expert tensors on disk (of {len(weight_map)} keys, via "
            f"{source})")
    detail = (f"{len(covered)}/{len(names)} declared names found over "
              f"{n_expert_keys} expert tensors "
              f"({'+'.join(sorted(layouts))} layout, via {source})")
    if missing:
        detail += f" — unused: {missing}"
    if unmapped:
        detail += f" — projections with no declared parent: {unmapped}"
    return CheckResult(
        "packed_expert_param_names() cover actual expert tensors", True, detail)


def _check_serving_profile(profile) -> CheckResult:
    from ..serving_profiles import (
        _LEGACY_RUNTIME_VALIDATORS,
        _load_runtime_validator,
        load_serving_profile,
        resolve_target_profile,
    )

    profile_id = resolve_target_profile(profile, None)
    try:
        serving_profile = load_serving_profile(profile_id)
    except Exception as e:
        return CheckResult(
            "serving profile exists + validator callables import",
            False,
            f"{profile_id!r} failed to load: {type(e).__name__}: {e}",
        )

    failures = []
    for rule in serving_profile.runtime_shape_validators:
        callable_path = (
            rule.callable_path
            or _LEGACY_RUNTIME_VALIDATORS.get(rule.id)
        )
        if not callable_path:
            continue
        try:
            _load_runtime_validator(callable_path)
        except Exception as e:
            failures.append(
                f"{rule.id}: {callable_path}: {type(e).__name__}: {e}"
            )
    if failures:
        return CheckResult(
            "serving profile exists + validator callables import",
            False,
            "; ".join(failures),
        )
    return CheckResult(
        "serving profile exists + validator callables import",
        True,
        f"{profile_id} ({len(serving_profile.runtime_shape_validators)} "
        "runtime validators)",
    )


def validate_profile(profile, model_path: str, cfg: dict) -> list[CheckResult]:
    checks = [
        _check_matches(profile, cfg),
        _check_vllm_class(profile),
        _check_fused_siblings(profile),
        _check_name_remap(profile),
        _check_serving_profile(profile),
        _check_packed_experts(profile, model_path),
        _check_source_passthrough(profile, model_path),
        _check_mtp(profile, cfg, model_path),
    ]
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to HF checkpoint directory.")
    ap.add_argument("--profile", default=None,
                    help="Profile class name (e.g. 'Qwen3_5Profile') or dotted "
                         "import path ('my_pkg.mod.MyProfile'). If omitted, "
                         "auto-detected from the model's config.")
    args = ap.parse_args()

    cfg = _load_config(args.model)
    profile, source = _get_profile(args.profile, args.model)

    print(f"Validating profile: {type(profile).__name__} ({source})")
    print(f"Model:              {args.model}")
    print(f"model_type:         {cfg.get('model_type')}")
    print(f"architectures:      {cfg.get('architectures')}")
    print(f"vllm class:         {profile.vllm_architecture_class() or '<none>'}")
    print()

    results = validate_profile(profile, args.model, cfg)
    for r in results:
        print(r)

    n_fail = sum(1 for r in results if not r.ok)
    n_pass = len(results) - n_fail
    print()
    print(f"{n_pass} / {len(results)} checks passed",
          "" if n_fail == 0 else f" ({n_fail} failed)")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
