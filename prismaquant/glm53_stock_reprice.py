"""GLM-5.3-Flash driver for the stock-vLLM anchored cost lane.

Pairs ``prismaquant.stock_anchored_cost`` (the platform mapping plugin) with
this one checkpoint.  Two modes matter:

``inventory``
    CPU-only, ~seconds, reads nothing but metadata.  Reports every admission
    and every blocker for the GLM campaign and returns success even when
    blocked, so a blocker is a *finding* on a report rather than a crash.  It
    refuses to substitute a resident entrypoint to make itself pass: a 598.5
    GiB bf16-resident load cannot fit a 121 GiB pool, so any inventory that
    "passes" by pointing at the resident path is passing a test the campaign
    will fail.

``run_stock_anchor_campaign``
    The GPU campaign entry.  Orchestration only -- it plans anchors, wraps
    measured production scalars, prices, merges and writes.  It does not
    contain a renderer: see ``ANCHOR_MEASUREMENT_CONTRACT`` for why the
    AURA adjoint cannot be re-derived per unit, and what must be supplied.

WHY THIS DRIVER EXISTS AT ALL
-----------------------------
When it was written (2026-08-26) the stock streamed AURA path was blocked at
that commit, and both legs were reproduced from the campaign's own receipts
(``/home/rob/dq-runs/glm53-flash/.cost_feasibility.json``):

  * ``aura_cost`` refused ``--checkpoint-dir`` without a value-bearing CB
    ``ProductionWeightCache`` identity, and ``cb_provenance`` is only
    populated when the menu holds a CB format.  The GLM menu holds none.
    **Resolved 2026-08-27**: the streamed checkpoint guard now accepts the
    production-anchor renderer's exact identity as the value-bearing render
    identity, so an anchored non-CB run checkpoints on the anchor identity
    alone (``tests/test_streamed_cost_checkpoints.py``).
  * ``--require-production-cache`` needs a retained dW cache, and
    ``--render-scope format-menu`` never retains one
    (``build_production_cache.py:394``).  **Moot under the anchored route**:
    the anchor renderer renders each layer transiently by design
    (``production_anchor_no_full_menu_materialization``), so no retained
    306B menu cache is required or wanted.

What remains of this driver is therefore not a workaround but the campaign
seam itself: ``prismaquant.glm53_stock_harvest`` runs the batched streamed
KL-adjoint harvest on the GPU box, and this module's ``campaign`` action
prices, merges and writes the allocator payload from its finished scalars
(``ANCHOR_MEASUREMENT_CONTRACT``), the expert-empirical rows, and the pinned
source terminals.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import pickle
import re
import sys

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    ANCHORED_AURA_COST_CURRENCY,
    ANCHORED_AURA_COST_SOURCE,
    cost_entry_is_anchored_aura_supersurrogate,
    cost_entry_is_bit_exact,
    serialized_candidate_payload,
)
from prismaquant.anchored_cost import (
    anchors_from_results,
    plan_anchor_requests,
    run_scalar_render_campaign,
)
from prismaquant.stock_anchored_cost import (
    DEFAULT_COSTED_FORMAT,
    RENDER_LEVERS,
    LadderRefusal,
    StockAnchoredCostError,
    StockAnchoredFormatPlugin,
    StockUnitDeclaration,
    anchors_from_measured_scalars,
    assert_probe_coverage,
    build_stock_allocator_cost_payload,
    build_stock_units,
    check_declaration_legality,
    expert_empirical_rows,
    merge_cost_rows,
    pinned_passthrough_rows,
    price_single_rung_candidates,
)

MODEL_PROFILE = "glm5_next"
SERVING_PROFILE = "vllm_glm5_next_packed_moe"
PROBE_CENSUS_SCHEMA = "prismaquant.stock_anchored.probe_census.v1"
CHECKPOINT_CENSUS_SCHEMA = "prismaquant.stock_anchored.checkpoint_census.v1"

#: The serving rule, copied from the profile spec so inventory classifies with
#: the same string the gate enforces. It is asserted against the loaded
#: serving profile rather than trusted (``_assert_rule_matches_profile``).
QUANTIZABLE_RE = re.compile(
    r"(^|[.])mlp[.](experts|shared_experts)([.]|$)"
    r"|(^|[.])mlp[.](gate_proj|up_proj|down_proj)([.]|$)"
)

#: Measured on the real checkpoint, quoted from the campaign's own
#: feasibility receipt. Present so the resident-substitution refusal can cite
#: a number instead of an opinion.
BF16_RESIDENT_GIB = 598.5465285144746
USABLE_GIB = 121.0

ANCHOR_MEASUREMENT_CONTRACT = (
    "AURA gW is a live-autograd quantity: aura_cost harvests weight.grad "
    "inside a post-accumulate hook and nulls it immediately, so no per-unit "
    "adjoint exists on disk that a standalone render(request) could "
    "re-derive. Computing one KL-adjoint backward per unit would multiply "
    "probe cost by the unit count. The batched harvest therefore stays "
    "batched, and the campaign consumes its finished scalars through "
    "anchors_from_measured_scalars -- exactly the interface cb_anchored_cost "
    "uses (anchors_from_streamed_payload). A measured row must carry "
    "dw_source='production_render' and production_anchor_measured=True."
)


class Glm53StockError(RuntimeError):
    """A GLM stock-lane inventory, identity, or orchestration refusal."""


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    path: str | None = None


# --------------------------------------------------------------------------
# Censuses
# --------------------------------------------------------------------------
def build_checkpoint_census(model_path: str | Path) -> dict[str, object]:
    """Classify every checkpoint tensor from the index alone.

    Index-only on purpose.  Source precision here is *per tensor, not per leaf
    name* -- ``o_proj`` is FP8 on the 12 MLA layers and BF16 on the 34 KDA
    layers -- so any leaf-name-keyed dtype map is wrong for this checkpoint.
    The profile spec's own instruction is to key on the presence of the
    ``.weight_scale_inv`` sibling, which the index answers exactly and in one
    8 MB read rather than 62 shard-header reads over NFS.
    """
    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise Glm53StockError(
            f"{index_path} is absent; refusing to guess the source layout"
        )
    weight_map = json.loads(index_path.read_text()).get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise Glm53StockError(f"{index_path} has no weight_map")
    keys = set(weight_map)
    config = json.loads((root / "config.json").read_text())

    from prismaquant.model_profiles.glm5_next import Glm5NextProfile

    profile = Glm5NextProfile()
    packed_by_projection: dict[str, str] = {}
    for packed_param in sorted(profile.packed_expert_param_names()):
        for projection in profile.packed_expert_projection_names(packed_param):
            packed_by_projection[str(projection)] = str(packed_param)

    per_expert = re.compile(
        r"^(?P<parent>.*[.]mlp[.]experts)[.]\d+[.](?P<proj>[a-z_]+)$"
    )
    units: dict[str, dict[str, object]] = {}
    unresolved: list[str] = []
    for key in sorted(keys):
        if not key.endswith(".weight"):
            continue
        live = profile.checkpoint_to_live_name(key)
        if live is None:
            continue
        stem = live[: -len(".weight")]
        match = per_expert.match(stem)
        if match is not None:
            packed = packed_by_projection.get(match.group("proj"))
            if packed is None:
                unresolved.append(key)
                continue
            unit = f"{match.group('parent')}.{packed}"
        else:
            unit = stem
        source_kind = (
            "fp8" if f"{key}_scale_inv" in keys else "bf16"
        )
        row = units.setdefault(unit, {
            "source_kinds": set(),
            "member_count": 0,
            "witness": key,
        })
        row["source_kinds"].add(source_kind)
        row["member_count"] = int(row["member_count"]) + 1
    mixed = sorted(
        unit for unit, row in units.items() if len(row["source_kinds"]) != 1
    )
    return {
        "schema": CHECKPOINT_CENSUS_SCHEMA,
        "model": str(root),
        "model_type": str(config.get("model_type") or ""),
        "architectures": [str(a) for a in (config.get("architectures") or [])],
        "index_entries": len(keys),
        "units": {
            unit: {
                "source_kind": sorted(row["source_kinds"])[0],
                "member_count": int(row["member_count"]),
                "witness": str(row["witness"]),
            }
            for unit, row in sorted(units.items())
            if len(row["source_kinds"]) == 1
        },
        "mixed_source_units": mixed,
        "unresolved_keys": sorted(unresolved)[:64],
        "unresolved_count": len(unresolved),
    }


def load_census(path: str | Path, *, schema: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise Glm53StockError(
            f"{path} is not a {schema} document; refusing to read a census "
            "whose contract is unknown"
        )
    return dict(payload)


# --------------------------------------------------------------------------
# Ladder construction from the two censuses
# --------------------------------------------------------------------------
def _shape_of(row: Mapping[str, object]) -> tuple[int, ...]:
    out = row.get("out_features")
    inp = row.get("in_features")
    if out is None or inp is None:
        raise Glm53StockError("probe row has no in/out features")
    experts = row.get("num_experts")
    if experts:
        return (int(experts), int(out), int(inp))
    return (int(out), int(inp))


def _role_of(qname: str) -> str:
    return qname.rsplit(".", 1)[-1]


def _unit_class_of(row: Mapping[str, object]) -> str:
    return "packed_expert" if row.get("is_packed") else "dense"


def build_declarations(
    probe_census: Mapping[str, object],
    checkpoint_census: Mapping[str, object],
    *,
    costed_format: str = DEFAULT_COSTED_FORMAT,
) -> tuple[
    list[StockUnitDeclaration],
    dict[str, tuple[str, int]],
    list[LadderRefusal],
    list[str],
]:
    """Split the probe universe into costed ladders, pins, and refusals.

    Returns ``(declarations, pinned, refusals, unresolved)``.  Every probe
    unit lands in exactly one of the four, and inventory asserts that: a unit
    that silently belonged to none would be a unit the allocator never prices
    and nothing reports.
    """
    ck_units = checkpoint_census.get("units")
    if not isinstance(ck_units, Mapping):
        raise Glm53StockError("checkpoint census has no units")
    probe_units = probe_census.get("units")
    if not isinstance(probe_units, Mapping):
        raise Glm53StockError("probe census has no units")

    declarations: list[StockUnitDeclaration] = []
    pinned: dict[str, tuple[str, int]] = {}
    refusals: list[LadderRefusal] = []
    unresolved: list[str] = []

    for qname in sorted(probe_units):
        row = probe_units[qname]
        entry = ck_units.get(qname)
        if not isinstance(entry, Mapping):
            unresolved.append(qname)
            continue
        source_kind = str(entry["source_kind"])
        terminal = "FP8_SOURCE" if source_kind == "fp8" else "BF16"
        shape = _shape_of(row)
        n_params = int(row["n_params"])
        if not QUANTIZABLE_RE.search(qname):
            # Profile-pinned: the serving profile denies every quantized rung
            # here, so there is no ladder and no decision -- only the source
            # bytes, priced at their true zero.
            payload, _, _ = serialized_candidate_payload(
                fr.get_format(terminal), shape,
                qname=qname, cb_serialization_context=None,
            )
            pinned[qname] = (terminal, int(payload))
            continue
        payload_bytes: dict[str, int] = {}
        for name in (costed_format, terminal):
            payload, _, _ = serialized_candidate_payload(
                fr.get_format(name), shape,
                qname=qname, cb_serialization_context=None,
            )
            payload_bytes[name] = int(payload)
        declaration = StockUnitDeclaration(
            qname=qname,
            role=_role_of(qname),
            unit_class=_unit_class_of(row),
            n_params=n_params,
            source_kind=source_kind,
            costed_format=costed_format,
            terminal_format=terminal,
            payload_bytes_by_format=payload_bytes,
        )
        found = check_declaration_legality(
            declaration, shape=shape, target_profile=SERVING_PROFILE,
        )
        if found:
            refusals.extend(found)
            continue
        declarations.append(declaration)
    return declarations, pinned, refusals, unresolved


def _assert_rule_matches_profile() -> None:
    """Assert the classifier regex is the serving profile's own rule.

    The regex above is a *copy*.  A copy that drifts from the gate it mirrors
    is how a unit gets classified as pinned by inventory and as quantizable by
    export, so it is checked against the loaded profile rather than trusted.
    """
    from prismaquant.serving_profiles import load_serving_profile

    profile = load_serving_profile(SERVING_PROFILE)
    for rule in profile.format_rules:
        if rule.when.not_regex == QUANTIZABLE_RE.pattern:
            return
    raise Glm53StockError(
        f"no rule in {SERVING_PROFILE} carries the quantizable regex this "
        "driver classifies with; the copy has drifted from the gate"
    )


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------
def _campaign_provider_check(repo: Path) -> Check:
    """Require a concrete, wired campaign provider -- statically.

    Parsed, never imported: inventory must stay CPU-only, and importing the
    campaign merely to ask whether a symbol is callable would drag torch and a
    device context into a metadata-only check.  Parsing also rejects the
    false-positive state where a CLI shim exists but delegates to a symbol
    that was never implemented.

    This is the refusal in "refuses to substitute resident entrypoints to make
    itself pass": the resident AURA path cannot run on this checkpoint
    (598.5 GiB bf16 resident against a 121 GiB pool), so an inventory that
    reported PASS because some resident entrypoint exists would be certifying
    a path the campaign cannot take.
    """
    worker = repo / "prismaquant/glm53_stock_reprice.py"
    if not worker.is_file():
        return Check(
            "stock campaign provider", "BLOCK",
            "driver seam prismaquant.glm53_stock_reprice is not implemented; "
            "refusing to substitute resident entrypoints",
            str(worker),
        )
    try:
        tree = ast.parse(worker.read_text())
    except SyntaxError as exc:
        return Check(
            "stock campaign provider", "BLOCK",
            f"driver does not parse: {exc}", str(worker),
        )
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    provider = functions.get("run_stock_anchor_campaign")
    if provider is None:
        return Check(
            "stock campaign provider", "BLOCK",
            "driver defines no top-level run_stock_anchor_campaign", str(worker),
        )
    body = [
        node for node in provider.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    ]
    if not body or all(
        isinstance(node, (ast.Pass, ast.Raise)) for node in body
    ):
        return Check(
            "stock campaign provider", "BLOCK",
            "run_stock_anchor_campaign is only a placeholder, not "
            "orchestration", str(worker),
        )
    calls = {
        node.func.id for node in ast.walk(provider)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    required = {
        "build_stock_units", "plan_anchor_requests",
        "price_single_rung_candidates", "merge_cost_rows",
        "build_stock_allocator_cost_payload",
    }
    missing = sorted(required - calls)
    if missing:
        return Check(
            "stock campaign provider", "BLOCK",
            f"run_stock_anchor_campaign does not invoke {missing}", str(worker),
        )
    return Check(
        "stock campaign provider", "PASS",
        "driver invokes the concrete anchored pipeline; no resident "
        "entrypoint is substituted", str(worker),
    )


def _admission_selftest() -> Check:
    """Prove the emitted row shape lands in the allocator branch it targets.

    A schema table in a report is a claim; running the allocator's own
    predicates over a representative row is evidence.  Both directions are
    checked, because the two row shapes this lane emits are priced by
    different branches and a near-miss in either is silent.
    """
    anchored = {
        "predicted_dloss": 4.0e-6,
        "memory_bytes": 1024,
        "cost_currency": ANCHORED_AURA_COST_CURRENCY,
        "cost_source": ANCHORED_AURA_COST_SOURCE,
        "fisher_application_count": 1,
    }
    if not cost_entry_is_anchored_aura_supersurrogate(anchored):
        return Check(
            "allocator admission", "BLOCK",
            "the anchored row shape this lane emits is not admitted by "
            "cost_entry_is_anchored_aura_supersurrogate", None,
        )
    terminal = {
        "predicted_dloss": 0.0, "weight_mse": 0.0,
        "output_mse": 0.0, "output_mse_measured": False,
    }
    if not cost_entry_is_bit_exact(terminal, "FP8_SOURCE"):
        return Check(
            "allocator admission", "BLOCK",
            "the FP8_SOURCE terminal row is not exact-by-construction to the "
            "allocator", None,
        )
    return Check(
        "allocator admission", "PASS",
        "anchored rows admit via cost_entry_is_anchored_aura_supersurrogate "
        "(currency+production_arm_render+fisher_application_count==1); "
        "terminals admit via cost_entry_is_bit_exact", None,
    )


def inventory_checks(
    *,
    repo: Path,
    model: Path,
    probe_census: Mapping[str, object] | None,
    probe_census_path: Path | None,
    probe_census_error: str | None,
    checkpoint_census: Mapping[str, object],
    expert_empirical: Path,
    checkpoint_dir: Path,
) -> list[Check]:
    checks: list[Check] = [
        Check(
            "gpu-free", "PASS",
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')!r}; "
            "inventory reads metadata only", None,
        ),
        _campaign_provider_check(repo),
        _admission_selftest(),
    ]

    # -- resident substitution refusal -------------------------------------
    checks.append(Check(
        "resident load refusal", "PASS",
        f"bf16 resident {BF16_RESIDENT_GIB:.1f} GiB exceeds the {USABLE_GIB:.0f} "
        "GiB unified pool by "
        f"{BF16_RESIDENT_GIB - USABLE_GIB:.1f} GiB, so streaming is mandatory "
        "and no resident fallback is offered by this driver", None,
    ))

    # -- checkpoint --------------------------------------------------------
    model_type = str(checkpoint_census.get("model_type") or "")
    if model_type != MODEL_PROFILE:
        checks.append(Check(
            "checkpoint census", "BLOCK",
            f"model_type {model_type!r} is not {MODEL_PROFILE!r}", str(model),
        ))
    else:
        checks.append(Check(
            "checkpoint census", "PASS",
            f"model_type={model_type} architectures="
            f"{checkpoint_census.get('architectures')} "
            f"index_entries={checkpoint_census.get('index_entries')} "
            f"resolved_units={len(checkpoint_census.get('units') or {})}",
            str(model),
        ))
    mixed = list(checkpoint_census.get("mixed_source_units") or [])
    unresolved_keys = int(checkpoint_census.get("unresolved_count") or 0)
    checks.append(Check(
        "source-kind resolution",
        "PASS" if not mixed and not unresolved_keys else "BLOCK",
        (
            "every resolved unit has one source precision"
            if not mixed and not unresolved_keys
            else f"{len(mixed)} mixed-precision unit(s) {mixed[:3]}; "
                 f"{unresolved_keys} unmapped weight key(s)"
        ),
        None,
    ))

    # -- serving profile ---------------------------------------------------
    try:
        _assert_rule_matches_profile()
        checks.append(Check(
            "serving profile rule", "PASS",
            f"{SERVING_PROFILE} carries the quantizable regex this driver "
            "classifies with", None,
        ))
    except Exception as exc:
        checks.append(Check(
            "serving profile rule", "BLOCK", str(exc), None,
        ))

    # -- probe -------------------------------------------------------------
    if probe_census is None:
        checks.append(Check(
            "probe census", "BLOCK",
            probe_census_error or "no probe census supplied",
            str(probe_census_path) if probe_census_path else None,
        ))
        return checks
    meta = probe_census.get("meta") or {}
    identity = probe_census.get("probe") or {}
    units = probe_census.get("units") or {}
    packed = sum(1 for row in units.values() if row.get("is_packed"))
    marginals = sum(1 for row in units.values() if row.get("has_marginals"))
    missing_norm = [
        name for name, row in units.items()
        if row.get("has_h_trace_raw") and row.get("h_trace_norm_tokens") is None
    ]
    checks.append(Check(
        "probe census", "PASS",
        f"n_stats={probe_census.get('n_stats')} packed={packed} "
        f"marginals={marginals} calib_hash={meta.get('calib_hash')} "
        f"nsamples={meta.get('nsamples')} seqlen={meta.get('seqlen')} "
        f"host={identity.get('host')} sha256={str(identity.get('sha256'))[:16]}",
        str(identity.get("path") or ""),
    ))
    checks.append(Check(
        "fisher renormalization stamp",
        "PASS" if not missing_norm else "BLOCK",
        (
            "every row carrying h_trace_raw also carries "
            "h_trace_norm_tokens, so allocator.renormalize_probe_fisher "
            "cannot SystemExit on a missing norm stamp"
            if not missing_norm
            else f"{len(missing_norm)} row(s) carry h_trace_raw with no "
                 f"h_trace_norm_tokens, e.g. {missing_norm[:3]}"
        ),
        None,
    ))

    # -- ladders -----------------------------------------------------------
    try:
        declarations, pinned, refusals, unresolved = build_declarations(
            probe_census, checkpoint_census,
        )
    except Exception as exc:
        checks.append(Check("ladder construction", "BLOCK", str(exc), None))
        return checks

    total = len(units)
    accounted = len(declarations) + len(pinned) + len(
        {item.qname for item in refusals}
    ) + len(unresolved)
    checks.append(Check(
        "unit accounting",
        "PASS" if accounted == total else "BLOCK",
        f"{len(declarations)} costed + {len(pinned)} pinned + "
        f"{len({item.qname for item in refusals})} refused + "
        f"{len(unresolved)} unresolved = {accounted} of {total} probe units",
        None,
    ))

    by_reason: dict[tuple[str, str, str], list[str]] = {}
    for item in refusals:
        by_reason.setdefault(
            (item.kind, item.format_name, item.reason), []
        ).append(item.qname)
    for (kind, format_name, reason), names in sorted(by_reason.items()):
        checks.append(Check(
            f"ladder refusal: {kind}/{format_name}", "BLOCK",
            f"{len(names)} unit(s) refused with reason={reason}; "
            f"e.g. {sorted(names)[:2]}",
            None,
        ))
    if not refusals:
        checks.append(Check(
            "ladder legality", "PASS",
            f"all {len(declarations)} costed units carry a legal "
            f"{DEFAULT_COSTED_FORMAT} rung and a legal source terminal", None,
        ))

    if unresolved:
        checks.append(Check(
            "probe/checkpoint reconciliation", "BLOCK",
            f"{len(unresolved)} probe unit(s) have no checkpoint tensor, "
            f"e.g. {unresolved[:3]}", None,
        ))
    else:
        checks.append(Check(
            "probe/checkpoint reconciliation", "PASS",
            "every probe unit resolves to checkpoint tensors", None,
        ))

    # The complement of the check above, and the one that actually catches a
    # silent undershoot: a checkpoint unit the serving rule calls quantizable
    # but the probe never saw would be priced by nobody, allocated by nobody,
    # and -- without this -- reported by nobody. A byte budget computed over
    # such a table is short by exactly the mass nothing mentioned.
    ck_quantizable = sorted(
        name for name in (checkpoint_census.get("units") or {})
        if QUANTIZABLE_RE.search(name)
    )
    unprobed = [name for name in ck_quantizable if name not in units]
    checks.append(Check(
        "checkpoint/probe complement",
        "PASS" if not unprobed else "BLOCK",
        (
            f"all {len(ck_quantizable)} quantizable checkpoint unit(s) are "
            "present in the probe; no quantizable mass is unpriced"
            if not unprobed
            else f"{len(unprobed)} quantizable checkpoint unit(s) were never "
                 f"probed, e.g. {unprobed[:3]}; their mass would be allocated "
                 "by nobody and reported by nobody"
        ),
        None,
    ))

    # -- promotable mass, given the refusals above --------------------------
    costed_params = sum(int(item.n_params) for item in declarations)
    refused_params = sum(
        int(units[name]["n_params"])
        for name in {item.qname for item in refusals}
        if name in units
    )
    quantizable_params = costed_params + refused_params
    if quantizable_params:
        share = 100.0 * costed_params / quantizable_params
        checks.append(Check(
            "promotable mass",
            "PASS" if not refusals else "BLOCK",
            f"{costed_params:,} of {quantizable_params:,} quantizable params "
            f"({share:.2f}%) have a legal costed rung AND a legal terminal, "
            f"so only that share can trade up from {DEFAULT_COSTED_FORMAT}; "
            f"the refused {refused_params:,} params are pinned to "
            f"{DEFAULT_COSTED_FORMAT} by the ladder refusals above",
            None,
        ))

    # -- units the campaign can actually build -----------------------------
    try:
        plugin = StockAnchoredFormatPlugin(
            arm_identity={"render_levers": dict(RENDER_LEVERS)},
            serving_profile_id=SERVING_PROFILE,
        )
        specs = build_stock_units(declarations, plugin)
        requests = plan_anchor_requests(specs, plugin)
        checks.append(Check(
            "anchor plan", "PASS",
            f"{len(requests)} production-arm renders planned, one per costed "
            f"unit; menu = {{{DEFAULT_COSTED_FORMAT}, source terminal}}", None,
        ))
    except Exception as exc:
        checks.append(Check("anchor plan", "BLOCK", str(exc), None))

    # -- packed-expert empirical half --------------------------------------
    packed_names = sorted(
        name for name, row in units.items()
        if row.get("is_packed") and QUANTIZABLE_RE.search(name)
    )
    if expert_empirical.is_file():
        try:
            with expert_empirical.open("rb") as handle:
                rows = expert_empirical_rows(pickle.load(handle))
            missing = [name for name in packed_names if name not in rows]
            checks.append(Check(
                "packed-expert empirical unit-KL",
                "PASS" if not missing else "BLOCK",
                (
                    f"{len(rows)} unit(s) priced empirically; all "
                    f"{len(packed_names)} packed units covered"
                    if not missing
                    else f"{len(missing)} packed unit(s) unpriced, e.g. "
                         f"{missing[:3]}"
                ),
                str(expert_empirical),
            ))
        except Exception as exc:
            checks.append(Check(
                "packed-expert empirical unit-KL", "BLOCK",
                str(exc), str(expert_empirical),
            ))
    else:
        checks.append(Check(
            "packed-expert empirical unit-KL", "BUILD",
            f"absent; {len(packed_names)} packed routed-expert unit(s) have "
            "no serving-unit KL yet. The smooth AURA cost is route-flip-blind "
            "for routed experts, so this file is not optional",
            str(expert_empirical),
        ))

    # -- checkpoint dir ----------------------------------------------------
    resolved = checkpoint_dir.resolve()
    bad = (
        str(resolved).startswith("/tmp")
        or str(resolved).startswith(str(repo.resolve()))
    )
    checks.append(Check(
        "campaign checkpoint dir",
        "BLOCK" if bad else "PASS",
        (
            "resolves under /tmp or the repo; /tmp was cleared by an OOM and "
            "took a set of artifacts with it"
            if bad else "outside /tmp and outside the repo"
        ),
        str(resolved),
    ))
    return checks


def print_checks(checks: Sequence[Check]) -> None:
    print(f"GLM-5.3-Flash stock anchored-cost inventory: profile={MODEL_PROFILE} "
          f"serving={SERVING_PROFILE}")
    for check in checks:
        suffix = f" [{check.path}]" if check.path else ""
        print(f"{check.status:>5}  {check.name}: {check.detail}{suffix}")


# --------------------------------------------------------------------------
# The campaign
# --------------------------------------------------------------------------
def run_stock_anchor_campaign(
    *,
    declarations: Sequence[StockUnitDeclaration],
    pinned: Mapping[str, tuple[str, int]],
    plugin: StockAnchoredFormatPlugin,
    campaign_identity: Mapping[str, object],
    checkpoint_dir: str | Path,
    resume: bool,
    probe_stats_keys: Sequence[str],
    measured: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
    empirical: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
    expected_empirical_units: Sequence[str] = (),
    require_empirical: bool = True,
    refusals: Sequence[LadderRefusal] = (),
) -> dict[str, object]:
    """Plan, anchor, price, merge and build the allocator payload.

    Two anchor sources, one contract.  When ``measured`` is supplied the
    already-harvested production scalars are wrapped directly
    (``anchors_from_measured_scalars``) -- the interface the CB lane uses, and
    the one this checkpoint needs because the AURA adjoint is a live-autograd
    quantity (``ANCHOR_MEASUREMENT_CONTRACT``).  Otherwise the core's scalar
    campaign runs the plugin's renderer under the ``cost_stage_checkpoint``
    journal, which is what replaces the CB-identity requirement that blocks
    the streamed path.

    The journal keys on ``RenderRequest.request_id`` -- a hash of the semantic
    request, never a list position -- so a resumed run re-derives each stored
    receipt and compares semantics rather than trusting the file.
    """
    units = build_stock_units(declarations, plugin)
    requests = plan_anchor_requests(units, plugin)
    if measured is not None:
        anchors = anchors_from_measured_scalars(
            requests, measured,
            arm_identity=plugin.arm_identity,
            payload_identity=campaign_identity,
        )
    else:
        results = run_scalar_render_campaign(
            requests, plugin,
            checkpoint_dir=checkpoint_dir,
            identity=campaign_identity,
            resume=resume,
            stage="glm53-stock-anchored-render",
        )
        anchors = anchors_from_results(requests, results)
    anchored = price_single_rung_candidates(units, plugin, anchors)
    merged, report = merge_cost_rows(
        anchored=anchored,
        pinned=pinned_passthrough_rows(pinned),
        empirical=empirical,
        expected_empirical_units=expected_empirical_units,
        require_empirical=require_empirical,
    )
    unpriced = assert_probe_coverage(
        merged, {str(name): {} for name in probe_stats_keys},
    )
    return build_stock_allocator_cost_payload(
        costs=merged,
        merge_report=report,
        plugin=plugin,
        campaign_identity=campaign_identity,
        unpriced_probe_units=unpriced,
        refusals=refusals,
        extra_provenance={
            "anchor_measurement_contract": ANCHOR_MEASUREMENT_CONTRACT,
            "render_levers": dict(RENDER_LEVERS),
            "serving_profile_id": SERVING_PROFILE,
            "model_profile": MODEL_PROFILE,
        },
    )


def _quantizable_packed_units(
    probe_census: Mapping[str, object],
) -> list[str]:
    """Packed routed-expert units, from the probe census.

    Classified here rather than from surviving declarations because on this
    checkpoint every packed unit loses its FP8_SOURCE terminal to the serving
    profile (``profile_mismatch``) and lands in refusals before reaching a
    declaration.  Their price is the empirical serving-unit KL.
    """
    probe_units = probe_census.get("units")
    if not isinstance(probe_units, Mapping):
        raise Glm53StockError("probe census has no units")
    return sorted(
        str(qname) for qname, row in probe_units.items()
        if isinstance(row, Mapping) and row.get("is_packed")
    )


def run_campaign_from_artifacts(
    *,
    probe_census: Mapping[str, object],
    checkpoint_census: Mapping[str, object],
    harvest: Mapping[str, object],
    expert_payload: Mapping[str, object],
) -> dict[str, object]:
    """Price/merge the finished harvest + empirical + pinned rows (CPU).

    Every cross-artifact identity this function can check cheaply, it checks
    before pricing: the harvest must be the full-plan run for exactly the
    dense declarations these censuses produce, its rows must be production
    anchors (zero RTN fallbacks), and its calibration must be the probe's.
    """
    harvest_schema = "prismaquant.glm53_stock_harvest.v1"
    if harvest.get("schema") != harvest_schema:
        raise Glm53StockError(
            f"harvest schema {harvest.get('schema')!r} is not {harvest_schema}"
        )
    if harvest.get("plan_scope") != "full":
        raise Glm53StockError(
            f"harvest plan_scope {harvest.get('plan_scope')!r} is not 'full'; "
            "a filtered smoke harvest cannot price the campaign"
        )
    declarations, pinned, refusals, unresolved = build_declarations(
        probe_census, checkpoint_census,
    )
    if unresolved:
        raise Glm53StockError(
            f"{len(unresolved)} probe unit(s) unresolved, e.g. {unresolved[:5]}"
        )
    dense = [d for d in declarations if d.unit_class == "dense"]
    if len(dense) != len(declarations):
        raise Glm53StockError(
            "non-dense declarations survived the ladder; the packed-expert "
            "route through refusals has changed and this merge must be "
            "re-derived, not patched"
        )
    plan = harvest.get("plan")
    if not isinstance(plan, Mapping):
        raise Glm53StockError("harvest carries no plan")
    expected_plan = {d.qname: [DEFAULT_COSTED_FORMAT] for d in dense}
    got_plan = {str(q): list(f) for q, f in plan.items()}
    if got_plan != expected_plan:
        missing = sorted(set(expected_plan) - set(got_plan))
        extra = sorted(set(got_plan) - set(expected_plan))
        raise Glm53StockError(
            "harvest plan differs from the dense declarations: "
            f"missing={missing[:5]} extra={extra[:5]} (or per-unit formats "
            "differ); refusing to price a table against a different plan"
        )
    arm_identity = harvest.get("arm_identity")
    if not isinstance(arm_identity, Mapping):
        raise Glm53StockError("harvest carries no arm identity")
    for field, expected in (
        ("model_profile", MODEL_PROFILE),
        ("serving_profile_id", SERVING_PROFILE),
        ("costed_format", DEFAULT_COSTED_FORMAT),
        ("render_levers", dict(RENDER_LEVERS)),
    ):
        if arm_identity.get(field) != expected:
            raise Glm53StockError(
                f"harvest arm identity {field}={arm_identity.get(field)!r} "
                f"differs from this driver's {expected!r}"
            )
    probe_meta = probe_census.get("meta") or {}
    calibration = arm_identity.get("calibration") or {}
    if not isinstance(calibration, Mapping) or (
        calibration.get("probe_calib_hash") != probe_meta.get("calib_hash")
    ):
        raise Glm53StockError(
            "harvest calibration is not bound to this probe census's "
            f"calib_hash ({probe_meta.get('calib_hash')!r})"
        )
    aura_payload = harvest.get("aura_payload")
    if not isinstance(aura_payload, Mapping):
        raise Glm53StockError("harvest carries no AURA payload")
    provenance = aura_payload.get("provenance") or {}
    if int(provenance.get("dw_rtn_fallback_rows", -1)) != 0:
        raise Glm53StockError(
            "harvest AURA payload carries RTN-fallback rows; every anchor "
            "must be the production render the exporter ships"
        )
    if int(provenance.get("dw_production_anchor_rows", 0)) != len(dense):
        raise Glm53StockError(
            f"harvest carries {provenance.get('dw_production_anchor_rows')} "
            f"production-anchor row(s) for {len(dense)} dense unit(s)"
        )
    measured = aura_payload.get("costs")
    if not isinstance(measured, Mapping):
        raise Glm53StockError("harvest AURA payload carries no cost rows")
    packed_names = _quantizable_packed_units(probe_census)
    empirical = expert_empirical_rows(expert_payload)
    plugin = StockAnchoredFormatPlugin(
        arm_identity=arm_identity,
        serving_profile_id=SERVING_PROFILE,
    )
    expert_provenance = expert_payload.get("provenance") or {}
    campaign_identity = {
        "schema": "prismaquant.glm53_stock_campaign.v1",
        "harvest": {
            "git_commit": provenance.get("git_commit"),
            "calib_hash": provenance.get("calib_hash"),
            "n_probes": aura_payload.get("n_probes"),
            "checkpoint_dir": harvest.get("checkpoint_dir"),
        },
        "expert_empirical": {
            "schema": expert_payload.get("schema"),
            "git_commit": expert_provenance.get("git_commit"),
            "eval_driver": expert_provenance.get("eval_driver"),
            "calib_batch": expert_provenance.get("calib_batch"),
        },
        "probe_census_calib_hash": probe_meta.get("calib_hash"),
    }
    return run_stock_anchor_campaign(
        declarations=dense,
        pinned=pinned,
        plugin=plugin,
        campaign_identity=campaign_identity,
        checkpoint_dir=str(harvest.get("checkpoint_dir") or "measured"),
        resume=False,
        probe_stats_keys=sorted(probe_census.get("units") or {}),
        measured=measured,
        empirical=empirical,
        expected_empirical_units=packed_names,
        require_empirical=True,
        refusals=refusals,
    )


RECIPE_REKEY_SCHEMA = "prismaquant.glm53_stock_campaign.recipe_rekey.v1"

# Probe payload maps whose keys are Linear/router qnames. `stats` rows also
# carry a `router_path` VALUE (the owning router's qname) that must move with
# the keys or the expert<->router association silently dangles across the
# namespace boundary.
_PROBE_QNAME_MAPS = (
    "stats",
    "router_counts",
    "router_totals",
    "router_active_counts",
    "expert_route_stats",
    "expert_info",
)


def _recipe_qname_map(
    names: Sequence[str],
) -> dict[str, str]:
    """live->recipe qname map via the glm5_next structure spec, refused on
    any collision.

    glm5_next is the first profile where probe artifacts and the recipe
    namespace genuinely diverge: `requires_multimodal_skeleton()` forces
    every staging through the multimodal wrapper, so probe/harvest recorded
    LIVE qnames (`model.language_model.layers.N.self_attn.forget_gate.*`)
    while the allocator's entire downstream contract - source-dtype
    manifest, `is_pinned_name`, export's assignment lookups - speaks the
    spec's RECIPE namespace (`model.layers.N.self_attn.f_a_proj`). The
    rename is deterministic and value-free; a collision would mean two
    measured units merging into one row, so it is refused, never resolved.
    """
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile

    profile = Glm5NextProfile()
    mapping: dict[str, str] = {}
    seen: dict[str, str] = {}
    for name in names:
        recipe = str(profile.live_to_recipe_name(str(name)))
        if not recipe:
            raise Glm53StockError(
                f"live qname {name!r} mapped to an empty recipe name"
            )
        prior = seen.get(recipe)
        if prior is not None and prior != name:
            raise Glm53StockError(
                f"recipe-namespace collision: {prior!r} and {name!r} both "
                f"map to {recipe!r}; refusing to merge measured units"
            )
        seen[recipe] = str(name)
        mapping[str(name)] = recipe
    return mapping


def rekey_probe_to_recipe(probe: Mapping[str, object]) -> dict[str, object]:
    """Return a recipe-keyed copy of a live-keyed probe payload."""
    meta = probe.get("meta")
    if isinstance(meta, Mapping) and meta.get("recipe_rekey"):
        raise Glm53StockError(
            "probe payload already carries a recipe_rekey stamp; refusing "
            "to rekey twice"
        )
    all_names: set[str] = set()
    for map_name in _PROBE_QNAME_MAPS:
        table = probe.get(map_name)
        if isinstance(table, Mapping):
            all_names.update(str(k) for k in table)
    for row in (probe.get("stats") or {}).values():
        if isinstance(row, Mapping) and row.get("router_path"):
            all_names.add(str(row["router_path"]))
    mapping = _recipe_qname_map(sorted(all_names))
    renamed = sum(int(new != old) for old, new in mapping.items())
    out: dict[str, object] = dict(probe)
    for map_name in _PROBE_QNAME_MAPS:
        table = probe.get(map_name)
        if not isinstance(table, Mapping):
            continue
        rekeyed: dict[str, object] = {}
        for key, value in table.items():
            new_key = mapping[str(key)]
            if map_name == "stats" and isinstance(value, Mapping) \
                    and value.get("router_path"):
                value = dict(value)
                value["router_path"] = mapping[str(value["router_path"])]
            rekeyed[new_key] = value
        if len(rekeyed) != len(table):
            raise Glm53StockError(
                f"probe[{map_name!r}] rekey changed cardinality "
                f"{len(table)} -> {len(rekeyed)}"
            )
        out[map_name] = rekeyed
    new_meta = dict(meta) if isinstance(meta, Mapping) else {}
    new_meta["recipe_rekey"] = {
        "schema": RECIPE_REKEY_SCHEMA,
        "renamed_keys": renamed,
        "total_keys": len(mapping),
    }
    out["meta"] = new_meta
    return out


def rekey_costs_to_recipe(cost: Mapping[str, object]) -> dict[str, object]:
    """Return a recipe-keyed copy of a live-keyed campaign cost payload."""
    provenance = cost.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("recipe_rekey"):
        raise Glm53StockError(
            "cost payload already carries a recipe_rekey stamp; refusing "
            "to rekey twice"
        )
    costs = cost.get("costs")
    if not isinstance(costs, Mapping) or not costs:
        raise Glm53StockError("cost payload has no costs table to rekey")
    mapping = _recipe_qname_map(sorted(str(k) for k in costs))
    rekeyed = {mapping[str(k)]: v for k, v in costs.items()}
    if len(rekeyed) != len(costs):
        raise Glm53StockError(
            f"cost rekey changed cardinality {len(costs)} -> {len(rekeyed)}"
        )
    out: dict[str, object] = dict(cost)
    out["costs"] = rekeyed
    new_prov = dict(provenance) if isinstance(provenance, Mapping) else {}
    new_prov["recipe_rekey"] = {
        "schema": RECIPE_REKEY_SCHEMA,
        "renamed_keys": sum(
            int(mapping[str(k)] != str(k)) for k in costs
        ),
        "total_keys": len(mapping),
    }
    out["provenance"] = new_prov
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prismaquant.glm53_stock_reprice",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="action", required=True)

    census = sub.add_parser(
        "checkpoint-census",
        help="classify checkpoint tensors from the safetensors index (CPU)",
    )
    census.add_argument("--model", required=True)
    census.add_argument("--output", required=True)

    inv = sub.add_parser(
        "inventory",
        help="report every admission and blocker for the GLM campaign (CPU)",
    )
    inv.add_argument("--model", required=True)
    inv.add_argument("--probe-census", required=True)
    inv.add_argument("--checkpoint-census", default=None)
    inv.add_argument("--expert-empirical", required=True)
    inv.add_argument("--checkpoint-dir", required=True)
    inv.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    inv.add_argument("--json", action="store_true")
    inv.add_argument(
        "--strict", action="store_true",
        help="exit 2 when any check BLOCKs (inventory reports by default)",
    )

    camp = sub.add_parser(
        "campaign",
        help="price/merge the finished harvest + empirical + pinned rows "
             "into the allocator cost payload (CPU)",
    )
    camp.add_argument("--probe-census", required=True)
    camp.add_argument("--checkpoint-census", required=True)
    camp.add_argument(
        "--harvest", required=True,
        help="glm53_stock_harvest wrapper pkl (plan_scope must be 'full')",
    )
    camp.add_argument("--expert-empirical", required=True)
    camp.add_argument("--output", required=True)

    rekey = sub.add_parser(
        "rekey-recipe",
        help="translate live-keyed probe/cost payloads into the spec's "
             "recipe namespace for the allocator (CPU, value-free rename)",
    )
    rekey.add_argument("--probe", required=True)
    rekey.add_argument("--costs", required=True)
    rekey.add_argument("--output-probe", required=True)
    rekey.add_argument("--output-costs", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Defense in depth: every CLI action in this module is CPU-only, and an
    # inventory that quietly opened a GPU would be reporting on a machine
    # state it does not own. Applied at entry rather than import so the GPU
    # harvest (glm53_stock_harvest) can import the declaration builder and
    # constants above without inheriting a blinded device mask.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args = _build_parser().parse_args(argv)
    if args.action == "checkpoint-census":
        census = build_checkpoint_census(args.model)
        Path(args.output).write_text(json.dumps(census, indent=1, sort_keys=True))
        print(
            f"[glm53-stock] checkpoint census -> {args.output} "
            f"({len(census['units'])} units, "
            f"{census['index_entries']} index entries)"
        )
        return 0

    if args.action == "campaign":
        probe_census = load_census(
            args.probe_census, schema=PROBE_CENSUS_SCHEMA,
        )
        checkpoint_census = load_census(
            args.checkpoint_census, schema=CHECKPOINT_CENSUS_SCHEMA,
        )
        with open(args.harvest, "rb") as handle:
            harvest = pickle.load(handle)
        with open(args.expert_empirical, "rb") as handle:
            expert_payload = pickle.load(handle)
        payload = run_campaign_from_artifacts(
            probe_census=probe_census,
            checkpoint_census=checkpoint_census,
            harvest=harvest,
            expert_payload=expert_payload,
        )
        from prismaquant.cost_stage_checkpoint import atomic_write_bytes

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            output,
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
        )
        report = payload["provenance"]["merge_report"]
        side = output.with_suffix(".provenance.json")
        side.write_text(json.dumps({
            "schema": payload["schema"],
            "output": str(output),
            "merge_report": report,
            "ladder_refusals": len(payload["provenance"]["ladder_refusals"]),
            "unpriced_probe_units": payload["provenance"][
                "unpriced_probe_units"
            ],
            "campaign_identity": payload["provenance"]["campaign_identity"],
        }, indent=2, sort_keys=True) + "\n")
        print(
            f"[glm53-stock] campaign cost payload -> {output} "
            f"(anchored {report['anchored_units']}, "
            f"empirical {report['empirical_units']}, "
            f"pinned {report['pinned_units']}, "
            f"cells {report['total_cells']}; sidecar {side.name})"
        )
        return 0

    if args.action == "rekey-recipe":
        from prismaquant.cost_stage_checkpoint import atomic_write_bytes

        with open(args.probe, "rb") as handle:
            probe = pickle.load(handle)
        with open(args.costs, "rb") as handle:
            cost = pickle.load(handle)
        probe_out = rekey_probe_to_recipe(probe)
        cost_out = rekey_costs_to_recipe(cost)
        stats_keys = set(probe_out["stats"])
        cost_keys = set(cost_out["costs"])
        if stats_keys != cost_keys:
            raise Glm53StockError(
                "rekeyed probe stats and cost tables disagree: "
                f"{sorted(stats_keys ^ cost_keys)[:4]}"
            )
        for out_path, payload in (
            (Path(args.output_probe), probe_out),
            (Path(args.output_costs), cost_out),
        ):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(
                out_path,
                pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
            )
        stamp = probe_out["meta"]["recipe_rekey"]
        print(
            f"[glm53-stock] recipe rekey: {stamp['renamed_keys']}/"
            f"{stamp['total_keys']} keys renamed -> "
            f"{args.output_probe}, {args.output_costs}"
        )
        return 0

    repo = Path(args.repo)
    probe_census: dict[str, object] | None = None
    probe_error: str | None = None
    try:
        probe_census = load_census(args.probe_census, schema=PROBE_CENSUS_SCHEMA)
    except Exception as exc:
        probe_error = (
            f"{exc}. probe.pkl for this campaign lives on sparklina at "
            "/home/rob/dq-runs/glm53-flash/work/artifacts/probe.pkl and is "
            "not readable from sparky; produce the census there and copy the "
            "JSON back."
        )
    if args.checkpoint_census:
        checkpoint_census = load_census(
            args.checkpoint_census, schema=CHECKPOINT_CENSUS_SCHEMA,
        )
    else:
        checkpoint_census = build_checkpoint_census(args.model)
    checks = inventory_checks(
        repo=repo,
        model=Path(args.model),
        probe_census=probe_census,
        probe_census_path=Path(args.probe_census),
        probe_census_error=probe_error,
        checkpoint_census=checkpoint_census,
        expert_empirical=Path(args.expert_empirical),
        checkpoint_dir=Path(args.checkpoint_dir),
    )
    if args.json:
        print(json.dumps(
            {"checks": [asdict(check) for check in checks]},
            indent=2, sort_keys=True,
        ))
    else:
        print_checks(checks)
    blocked = any(check.status == "BLOCK" for check in checks)
    return 2 if (blocked and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
