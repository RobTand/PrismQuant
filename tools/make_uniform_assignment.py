#!/usr/bin/env python3
"""Emit a legal UNIFORM ``layer_config.json`` — one format for every quantizable unit.

**Why this exists.** The ENDPOINTS of the format-choice substitution ladder
(``docs/design/format_choice_4p5.md`` §5 Stage 2) are *pure* single-format
builds — pure ``NVFP4`` and pure ``FP8_CB_K40`` — so that
the total format effect on quality and serving time is measured with **zero
allocation confound**. Neither endpoint is shippable: a single-rung menu is the
sanctioned cost-model-isolation pattern only (``FP8 in every recipe``). This
tool changes no default and does not touch the allocator; it is a second entry
point that produces the *same* artifact (``layer_config.json``) the allocator
produces, under the *same* legality.

**Legality is not reimplemented.** Every verdict is delegated:

===========================================  ==========================================
question                                     owner
===========================================  ==========================================
which tensors are quantizable, and which     ``ModelProfile.build_model_graph``
units must share one format                  (``model_profiles/structure.py``)
is (shape, format) legal on this profile,    ``allocator_candidates.check_format_applicability``
including passthrough-source integrity
what dtype is each source tensor             ``allocator_candidates._scan_source_dtype_manifest``
which fused groups are incomplete            ``decision_units.incomplete_fused_group_members``
is the LM head a tied alias of the embedding ``tied_embeddings.lm_head_is_tied_alias``
which serving profile applies                ``serving_profiles.resolve_target_profile``
===========================================  ==========================================

**Exceptions land exactly where the allocator lands them.** Profile-pinned
names (``lm_head`` / ``embed_tokens``), a tied LM head, and the present members
of an INCOMPLETE fused-sibling group are *omitted* from the assignment — which
is precisely what makes export keep them as BF16 passthrough and put them on
the ignore list (``allocator.py`` ``incomplete_fused_group_dp_exclusions`` /
``tied_lm_head_dp_exclusions``: excluded names are dropped from the DP and
therefore absent from ``layer_config``). A unit the requested format cannot
legally take is DEMOTED whole to ``--fallback-format`` (default ``BF16``, itself
legality-checked, so a non-bf16 source can never be handed a synthesized BF16);
if the fallback is illegal too, the unit is omitted.

Usage::

    python3 -m tools.make_uniform_assignment \\
        --model /home/rob/models/Qwen3-0.6B \\
        --format FP8_CB_K40 \\
        --target-profile nvfp4_cb \\
        --out $WORK_DIR/artifacts/layer_config.json \\
        --report $WORK_DIR/artifacts/uniform_assignment_report.json
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    PASSTHROUGH_SOURCE_REQUIREMENTS,
    _scan_source_dtype_manifest,
    check_format_applicability,
)
from prismaquant.decision_units import incomplete_fused_group_members
from prismaquant.layer_config import (
    LAYER_CONFIG_META_KEY,
    canonicalize_assignment,
    strip_weight,
)
from prismaquant.schemas import validate_layer_config_payload
from prismaquant.serving_profiles import resolve_target_profile

META_SCHEMA = "prismaquant.layer_config_meta.v1"


# ---------------------------------------------------------------------------
# The uniform assignment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UniformAssignment:
    """A uniform per-Linear assignment plus the reason for every exception."""

    assignment: dict[str, str]
    excluded: dict[str, str] = field(default_factory=dict)
    demoted_units: tuple[str, ...] = ()
    unit_count: int = 0
    params_by_format: dict[str, int] = field(default_factory=dict)

    @property
    def achieved_bits(self) -> float:
        """bpp over the QUANTIZABLE parameters this assignment covers.

        Core principle 12: bpp is reported over quantizable parameters only —
        omitted names (pinned head/embed, tied head, incomplete fused groups)
        are the non-quantizable floor and are excluded from both sides of the
        ratio, exactly as the allocator's body bpp is.
        """
        total_params = sum(self.params_by_format.values())
        if not total_params:
            return 0.0
        bits = sum(
            fr.get_format(fmt).effective_bits * n
            for fmt, n in self.params_by_format.items()
        )
        return bits / total_params


def _shape_of(graph, recipe_param_name: str) -> tuple[int, ...]:
    tensor = graph.by_recipe_name().get(recipe_param_name)
    return tuple(int(d) for d in (tensor.shape if tensor is not None else ()))


def _numel(shape: Iterable[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def build_uniform_assignment(
    graph,
    fmt: str,
    *,
    profile,
    target_profile: str | None = None,
    source_kinds: Mapping[str, str] | None = None,
    fallback_format: str | None = "BF16",
    tied_head_names: Iterable[str] = (),
) -> UniformAssignment:
    """Assign ``fmt`` to every quantizable unit of ``graph`` that can take it.

    ``graph`` is a ``model_profiles.structure.ModelGraph``; its
    ``optimization_units()`` decomposition is the authority on which Linears
    must move together (fused siblings, packed experts), so uniformity is
    enforced at UNIT granularity — a unit is never split by a legality verdict.
    """
    requested_formats = [fmt]
    if fallback_format:
        requested_formats.append(fallback_format)
    resolved_formats = fr.require_producer_formats(
        requested_formats,
        where="new uniform assignment",
    )
    fmt_name = resolved_formats[0].name
    fallback_name = (
        resolved_formats[1].name if fallback_format else None
    )
    kinds = dict(source_kinds or {})
    tied = {strip_weight(str(n)) for n in tied_head_names}

    quantizable = {t.recipe_name: strip_weight(t.recipe_name)
                   for t in graph.quantizable_tensors()}
    incomplete = incomplete_fused_group_members(set(quantizable.values()), profile)

    assignment: dict[str, str] = {}
    excluded: dict[str, str] = {}
    demoted: list[str] = []
    params_by_format: dict[str, int] = {}
    unit_count = 0

    for unit in graph.optimization_units():
        members = [m for m in unit.members if m in quantizable]
        if not members:
            continue
        unit_count += 1
        qnames = {m: quantizable[m] for m in members}

        # Structural exceptions. A fused/packed unit is atomic: if ANY member
        # is structurally excluded the whole unit is (a half-quantized fused
        # group is the vLLM fused-load KeyError this rule exists to prevent).
        structural = {
            m: ("tied_lm_head" if qnames[m] in tied else "incomplete_fused_group")
            for m in members
            if qnames[m] in tied or qnames[m] in incomplete
        }
        if structural:
            reason = sorted(set(structural.values()))[0]
            for m in members:
                excluded[qnames[m]] = structural.get(m, f"unit_{reason}")
            continue

        chosen: str | None = None
        detail = ""
        for candidate in (fmt_name, fallback_name):
            if candidate is None:
                continue
            verdicts = {
                m: check_format_applicability(
                    _shape_of(graph, m),
                    candidate,
                    qname=qnames[m],
                    source_kind=kinds.get(qnames[m]),
                    target_profile=target_profile,
                )
                for m in members
            }
            bad = {m: v for m, v in verdicts.items() if not v.legal}
            if not bad:
                chosen = candidate
                break
            if candidate == fmt_name:
                first = sorted(bad)[0]
                detail = f"{bad[first].reason}: {bad[first].detail}"

        if chosen is None:
            for m in members:
                excluded[qnames[m]] = f"illegal_{fmt_name}_and_fallback ({detail})"
            continue
        if chosen != fmt_name:
            demoted.append(unit.id)

        for m in members:
            assignment[qnames[m]] = chosen
            params_by_format[chosen] = (
                params_by_format.get(chosen, 0) + _numel(_shape_of(graph, m)))

    return UniformAssignment(
        assignment=assignment,
        excluded=excluded,
        demoted_units=tuple(sorted(demoted)),
        unit_count=unit_count,
        params_by_format=params_by_format,
    )


def assert_assignment_legal(
    assignment: Mapping[str, str],
    graph,
    *,
    profile,
    target_profile: str | None = None,
    source_kinds: Mapping[str, str] | None = None,
) -> None:
    """Re-gate a FINAL assignment through the allocator's own legality.

    Independent of how the assignment was produced, this asserts the two
    properties the allocator's output is asserted on: (1) every (Linear,
    format) pair is legal on the resolved serving profile, including
    passthrough-source integrity, and (2) every fused-sibling / packed-expert
    serving unit carries exactly ONE format.
    """
    kinds = dict(source_kinds or {})
    by_qname = {strip_weight(t.recipe_name): t for t in graph.tensors}

    try:
        fr.require_producer_formats(
            sorted(set(assignment.values())),
            where="final uniform assignment",
        )
    except ValueError as exc:
        raise AssertionError(str(exc)) from None

    illegal: list[str] = []
    for qname, fmt in sorted(assignment.items()):
        tensor = by_qname.get(qname)
        if tensor is None:
            illegal.append(f"{qname}: not a tensor in the model graph")
            continue
        verdict = check_format_applicability(
            tuple(int(d) for d in tensor.shape),
            fmt,
            qname=qname,
            source_kind=kinds.get(qname),
            target_profile=target_profile,
        )
        if not verdict.legal:
            illegal.append(f"{qname}: {fmt} -> {verdict.reason}: {verdict.detail}")
    if illegal:
        raise AssertionError(
            "uniform assignment is not legal under target_profile="
            f"{target_profile!r}:\n  " + "\n  ".join(illegal[:10]))

    # Passthrough integrity, asserted explicitly (the allocator's
    # belt-and-suspenders check) so a missing source manifest cannot make the
    # verdict above vacuously true.
    violations = [
        f"{qname}: picked {fmt} but source is {kinds.get(qname)!r}"
        for qname, fmt in sorted(assignment.items())
        if fmt in PASSTHROUGH_SOURCE_REQUIREMENTS
        and kinds
        and kinds.get(qname) not in (None, PASSTHROUGH_SOURCE_REQUIREMENTS[fmt])
    ]
    if violations:
        raise AssertionError(
            "passthrough-integrity violation:\n  " + "\n  ".join(violations[:10]))

    mixed: list[str] = []
    for unit in graph.optimization_units():
        fmts = {assignment[q] for q in
                (strip_weight(m) for m in unit.members) if q in assignment}
        present = [strip_weight(m) for m in unit.members
                   if strip_weight(m) in assignment]
        if len(fmts) > 1:
            mixed.append(f"{unit.id}: {sorted(fmts)}")
        elif present and len(present) != len(unit.members):
            mixed.append(
                f"{unit.id}: partially assigned "
                f"({len(present)}/{len(unit.members)} members)")
    if mixed:
        raise AssertionError(
            "serving units must carry ONE format and be assigned whole:\n  "
            + "\n  ".join(mixed[:10]))


def layer_config_payload(
    result: UniformAssignment,
    *,
    meta: Mapping | None = None,
) -> dict:
    """Render the assignment as an allocator-compatible layer_config payload."""
    fr.require_producer_formats(
        sorted(set(result.assignment.values())),
        where="uniform layer_config payload",
    )
    payload: dict = {
        name: fr.get_format(fmt).autoround_config()
        for name, fmt in sorted(result.assignment.items())
    }
    payload[LAYER_CONFIG_META_KEY] = dict(meta or {})
    validate_layer_config_payload(payload, "<uniform assignment>")
    # Round-trip through the production parser: what export/recache/KL read
    # back must be exactly what we intended to write.
    parsed = canonicalize_assignment(payload)
    drift = {k for k in set(parsed) | set(result.assignment)
             if parsed.get(k) != result.assignment.get(k)}
    if drift:
        raise AssertionError(
            "layer_config round-trip drifted for: " + ", ".join(sorted(drift)[:10]))
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _meta_skeleton(model_path: str):
    """A meta-device skeleton of the model: names + shapes + module types, 0 bytes.

    Same idiom as ``export_native_compressed`` (``init_empty_weights`` +
    ``AutoModelForCausalLM.from_config``); no weights are read, so this is
    CPU/RAM-free and scales to any model size.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    from prismaquant.sensitivity_probe import stage_text_only

    try:
        from accelerate import init_empty_weights
    except ImportError:  # pragma: no cover - accelerate ships with the venv
        from contextlib import nullcontext as init_empty_weights

    from prismaquant.streaming_model import _mask_cuda_queries_during_meta_init

    staged = stage_text_only(model_path)
    config = AutoConfig.from_pretrained(staged, trust_remote_code=True)
    with _mask_cuda_queries_during_meta_init("[uniform]"):
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Emit a legal uniform (single-format) layer_config.json.")
    ap.add_argument("--model", required=True,
                    help="source HF model directory")
    ap.add_argument("--format", required=True,
                    help="the ONE format every quantizable unit gets "
                         "(e.g. NVFP4, FP8_CB_K40)")
    ap.add_argument("--out", required=True, help="output layer_config.json")
    ap.add_argument("--target-profile", default=None,
                    help="serving profile override; unset lets the "
                         "architecture spec's default_serving_profile win")
    ap.add_argument("--target-profile-default", default="vllm_packed_moe",
                    help="fallback when the architecture declares none "
                         "(run-pipeline.sh's TARGET_PROFILE_DEFAULT)")
    ap.add_argument("--fallback-format", default="BF16",
                    help="format for units the requested one cannot legally "
                         "take; empty string omits them instead")
    ap.add_argument("--report", default=None,
                    help="optional JSON report (exceptions, per-format counts)")
    args = ap.parse_args(argv)

    from prismaquant.model_profiles.registry import detect_profile
    from prismaquant.tied_embeddings import lm_head_is_tied_alias

    profile = detect_profile(args.model)
    target_profile = resolve_target_profile(
        profile, args.target_profile, default=args.target_profile_default)
    print(f"[uniform] model={args.model}")
    print(f"[uniform] profile={profile.name} target_profile={target_profile}")

    model = _meta_skeleton(args.model)
    graph = profile.build_model_graph(model)
    source_kinds = _scan_source_dtype_manifest(args.model, profile)

    tied_head: list[str] = []
    if lm_head_is_tied_alias(args.model, profile=profile):
        head = profile.lm_head_name()
        tied_head = [strip_weight(t.recipe_name) for t in graph.tensors
                     if strip_weight(t.recipe_name) == head
                     or strip_weight(t.recipe_name).endswith("." + head)]
        print(f"[uniform] tied LM head excluded (shares storage with the "
              f"input embedding): {tied_head}")

    result = build_uniform_assignment(
        graph,
        args.format,
        profile=profile,
        target_profile=target_profile,
        source_kinds=source_kinds,
        fallback_format=args.fallback_format or None,
        tied_head_names=tied_head,
    )
    assert_assignment_legal(
        result.assignment, graph,
        profile=profile,
        target_profile=target_profile,
        source_kinds=source_kinds,
    )

    fmt_name = fr.get_format(args.format).name
    achieved = result.achieved_bits
    meta = {
        "schema": META_SCHEMA,
        "target_profile": target_profile,
        "target_profile_requested": args.target_profile,
        "target_profile_default": str(args.target_profile_default),
        "target_bits": float(achieved),
        "achieved_bits": float(achieved),
        "uniform_format": fmt_name,
        "fallback_format": (fr.get_format(args.fallback_format).name
                            if args.fallback_format else None),
        "generator": "tools/make_uniform_assignment.py",
        "model_path": str(args.model),
        "model_profile": str(profile.name),
        "note": "single-format ISOLATION build (format_choice_4p5 §5 Stage 2 "
                "endpoint) — not a shippable menu",
    }
    payload = layer_config_payload(result, meta=meta)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    counts: dict[str, int] = {}
    for value in result.assignment.values():
        counts[value] = counts.get(value, 0) + 1
    reasons: dict[str, int] = {}
    for reason in result.excluded.values():
        key = reason.split(" (")[0]
        reasons[key] = reasons.get(key, 0) + 1

    print(f"[uniform] units={result.unit_count} "
          f"assigned={len(result.assignment)} omitted={len(result.excluded)}")
    for value, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {value:>14}: {n:>5} Linears")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  omitted [{reason}]: {n}")
    if result.demoted_units:
        print(f"[uniform] {len(result.demoted_units)} unit(s) demoted to the "
              f"fallback (sample: {list(result.demoted_units)[:4]})")
    print(f"[uniform] achieved_bits={achieved:.4f} bpp over "
          f"{sum(result.params_by_format.values()):,} quantizable params")
    print(f"[uniform] layer_config -> {out}")

    if args.report:
        report = {
            "meta": meta,
            "unit_count": result.unit_count,
            "assigned": len(result.assignment),
            "format_counts": counts,
            "params_by_format": result.params_by_format,
            "achieved_bits": achieved,
            "demoted_units": list(result.demoted_units),
            "excluded": result.excluded,
        }
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[uniform] report -> {rp}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
