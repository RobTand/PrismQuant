#!/usr/bin/env python3
"""Measure the UNMEASURED high-rate band of the E2M1 (4-bit) trellis ladder.

The sealed hull sweep stopped at body rate 2.25; the 2026-08-29 menu extension
carried it to 3.0.  `TCQ_E2M1_R256`'s own mathematical bound is
``mathematical_q256_bounds == (256, 1016)`` -> body rate 3.96875, because a
block may spend `bypass_rate` (4) at up to 248 of 256 positions so long as
`MIN_TRELLIS_STEPS` (8) remain genuinely coded.  Everything from 3.0 to 3.96875
was never measured, and that is exactly the band where a W4A4 trellis would have
to beat scalar NVFP4 (4.0 body + 0.5 plane = 4.5 bpw).

WHY A SIBLING DRIVER AND NOT `hull_sweep.py --extra-rates`:
`hull_sweep`'s checkpoint identity contains the whole `plan`, so adding rates
invalidates every checkpoint and re-funds all 25 CB rungs on 24 tensors (the
08-29 menu extension cost 3h15m for exactly that reason).  `cb_two_tier` encode
is ~80% of ladder cost and CB is already retired 24/24 across the overlapping
band, so re-buying it answers nothing.  This driver imports `hull_sweep` and
calls its OWN functions -- same corpus, same hashes, same plane snap, same
context, same alphabets, same schedule builder, same footprint accountant, same
scorer -- so the rows it writes are the rows `hull_sweep` would have written.

SELF-CHECK, NOT SELF-CERTIFICATION: rate 3.0 is re-measured here and compared
against the 08-29 row.  If this driver does not reproduce that row, its new
rows are not comparable to the published ladder and it refuses.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_HULL_ROOT = Path("/home/rob/dq-runs/trellis-hull-20260828")
sys.path.insert(0, str(LOCKED_HULL_ROOT))

import numpy as np
import torch

import hull_sweep as H
from atomic_publication import (
    PublicationError,
    atomic_checkpoint_json,
    exclusive_publication_claim,
    file_sha256,
    identity_sha256,
    publish_file_no_replace,
)
from isolated_glm_corpus import (
    load_active_glm_corpus_bound,
    read_bound_json,
)
from numeric_checkpoint_contract import (
    CheckpointContractError,
    validate_e2m1_checkpoint,
    validate_e2_published_control_arm,
)
from numeric_execution_contract import (
    NumericExecutionContractError,
    require_numeric_execution_environment,
    require_repo_commit,
)

W, C, P, S4, TF = H.W, H.C, H.P, H.S4, H.TF
import prismaquant.export_native_compressed as NVFP4_COMPARATOR  # noqa: E402
NVFP4_ACTIVATION_CONTRACT = NVFP4_COMPARATOR._nvfp4_activation_contract

EXPECTED_NVFP4_COMPARATOR_SHA256 = (
    "cec4e8f18d36f0c9b1f70cc69959bcf27449fe048fb8c612c488ea042326129a"
)
EXPECTED_NVFP4_ACTIVATION_CONTRACT_SHA256 = (
    "690d345b371ad99a8355e3e2e52b7220a1e6d0160c497dbaffe255647896bb27"
)

# --- corpus configuration --------------------------------------------------
# dsv4 = MXFP4-source DSv4 routed experts (21-27 distinct source values).
# bf16 = bf16-source Qwen3-4B DENSE MLP (~5100 distinct source values).  The
# bf16 corpus fixes the SOURCE-DTYPE confound and NOTHING else: it is dense
# Qwen3-4B, not MoE experts and not GLM.  Label it that way everywhere.
CORPUS_LABEL = {
    "dsv4": "DSv4 routed experts, MXFP4 source",
    "bf16": "Qwen3-4B DENSE MLP, bf16 source (NOT MoE, NOT GLM)",
    "glm": "GLM-5.3-Flash expert 0 + dense MLP, bf16 source",
}
CONTROL_RATE = 3.0
NEW_RATES = (3.25, 3.5, 3.75, 3.9375, 3.96875)
# 3.96875 is the family's mathematical ceiling but is NOT universally
# reachable: at 1016 q256 every 256-block must keep exactly MIN_TRELLIS_STEPS=8
# coded positions, leaving the global reverse-water-fill zero slack to move a
# bit between blocks, and on at least one corpus tensor the rebalance is
# infeasible.  A refusal is recorded per (tensor, rate), never crashed on, and
# 3.9375 is the highest rung reachable on all 24.
MID_RATES = (1.5, 2.0, 2.5)
PUBLISHED = Path("/home/rob/dq-runs/trellis-hull-20260828/"
                 "hull_results.menuext-20260829.json")
BF16_PUBLISHED = Path("/home/rob/dq-runs/trellis-bf16-20260829/"
                      "bf16_w4a4_results.json")
# On bf16 the campaign question is the coding gain at matched rate, so the
# rungs are the integer rates the coordinator named plus 2.5 to read the TREND
# (the gain RISES with rate on DSv4: 1.34 -> 1.88 -> 2.18, and whether that
# survives on a continuous-density source is what the estimate assumed).
BF16_RATES = (1.0, 2.0, 2.5, 3.0)
GLM_RATE_PLANS = {
    "scaffold": BF16_RATES,
    "high": NEW_RATES,
}
# Deterministic given the same encoder/inputs/device.  A torch/triton skew
# flips DISCRETE encode decisions and shows at ~1e-3, which no tolerance
# absorbs -- it is diagnosed, not tolerated.  Same bar hull_sweep uses.
CONTROL_RTOL = 1e-9
CELL_KEYS = frozenset({
    "shape", "numel", "population", "weighted_energy", "plain_energy",
    "two_tier_plane_sha256", "arms", "unreachable_rungs", "control",
})


def _atomic_json(path: Path, value: dict) -> None:
    atomic_checkpoint_json(path, value)


def _checkpoint_document(
    receipt: Mapping[str, object],
    per_tensor: Mapping[str, object],
    *,
    partial: bool,
) -> dict[str, object]:
    body: dict[str, object] = {
        "receipt": {
            **receipt,
            "partial": partial,
            "tensors_done": len(per_tensor),
        },
        "per_tensor": dict(per_tensor),
    }
    return {**body, "checkpoint_sha256": identity_sha256(body)}


def _strict_json_object(path: Path) -> dict[str, object]:
    def object_from_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(), object_pairs_hook=object_from_pairs)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"FATAL: invalid partial checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"FATAL: partial checkpoint {path} is not an object")
    return value


def _resume_partial(
    path: Path,
    *,
    receipt: Mapping[str, object],
    expected_tensors: Mapping[str, Mapping[str, object]],
    expected_controls: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict], float] | None:
    if not path.exists():
        return None
    document = _strict_json_object(path)
    if set(document) != {"receipt", "per_tensor", "checkpoint_sha256"}:
        raise SystemExit("FATAL: partial checkpoint members differ")
    body = {key: document[key] for key in ("receipt", "per_tensor")}
    if document["checkpoint_sha256"] != identity_sha256(body):
        raise SystemExit("FATAL: partial checkpoint self-digest differs")
    try:
        validate_e2m1_checkpoint(
            document, current_receipt=receipt,
            expected_tensors=expected_tensors,
            expected_controls=expected_controls,
        )
    except CheckpointContractError as exc:
        raise SystemExit(f"FATAL: partial checkpoint contract differs: {exc}") from exc
    saved_receipt = document["receipt"]
    per_tensor = document["per_tensor"]
    started = saved_receipt.get("started_at_unix_s")
    if not isinstance(started, (int, float)) or not math.isfinite(started):
        raise SystemExit("FATAL: partial checkpoint start time is invalid")
    return dict(per_tensor), float(started)


def _e2_replay_semantics(cell: Mapping[str, object]) -> dict[str, object]:
    """Return every cell claim except explicitly non-claim wall timing."""

    normalized = copy.deepcopy(dict(cell))
    arms = normalized.get("arms")
    if isinstance(arms, dict):
        for arm in arms.values():
            if isinstance(arm, dict):
                arm.pop("encode_seconds", None)
    return normalized


def _require_e2_replay_match(
    name: str, saved: Mapping[str, object], regenerated: Mapping[str, object],
) -> None:
    if _e2_replay_semantics(saved) != _e2_replay_semantics(regenerated):
        raise SystemExit(
            f"FATAL: {name}: saved checkpoint cell differs from deterministic "
            "replay; refusing reuse"
        )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("dsv4", "bf16", "glm"),
                    default="dsv4")
    ap.add_argument(
        "--glm-manifest", type=Path,
        help=("finalized trellis.bf16_corpus.v2 manifest; required for glm. "
              "The explicit path is part of the result provenance."),
    )
    ap.add_argument(
        "--glm-rate-plan", choices=tuple(GLM_RATE_PLANS), default="scaffold",
        help=("GLM-only rate set: scaffold preserves the 1/2/2.5/3 contract; "
              "high measures the near-four-bit band without rerunning it"),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-control-drift", action="store_true",
                    help="record a failed 3.0 control instead of refusing")
    return ap.parse_args()


def _repo_commit() -> str:
    try:
        return require_repo_commit(REPO_ROOT)
    except NumericExecutionContractError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc


def _execution_environment() -> dict[str, object]:
    try:
        return require_numeric_execution_environment(
            REPO_ROOT,
            H.current_env(),
            os.environ,
            require_cuda=True,
        )
    except NumericExecutionContractError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc


def _active_source_identity() -> dict[str, object]:
    driver = Path(__file__).resolve(strict=True)
    isolated_loader = driver.with_name("isolated_glm_corpus.py")
    execution_contract = driver.with_name("numeric_execution_contract.py")
    corpus_reader = REPO_ROOT / "prismaquant/trellis_bf16_corpus.py"
    imported_codec_modules = {
        name: {
            "path": str(Path(module.__file__).resolve(strict=True)),
            "sha256": file_sha256(Path(module.__file__).resolve(strict=True)),
        }
        for name, module in (
            ("hull_sweep", H),
            ("weight_codec", W),
            ("common", C),
            ("plane", P),
            ("schedule", S4),
            ("trellis_formats", TF),
            ("nvfp4_scalar_comparator", NVFP4_COMPARATOR),
            ("nvfp4_activation_contract", NVFP4_ACTIVATION_CONTRACT),
        )
    }
    for module, expected_sha in (
        (NVFP4_COMPARATOR, EXPECTED_NVFP4_COMPARATOR_SHA256),
        (
            NVFP4_ACTIVATION_CONTRACT,
            EXPECTED_NVFP4_ACTIVATION_CONTRACT_SHA256,
        ),
    ):
        module_path = Path(module.__file__).resolve(strict=True)
        module_sha = file_sha256(module_path)
        if (not module_path.is_relative_to(H.SNAPSHOT.resolve(strict=True))
                or module_sha != expected_sha):
            raise SystemExit(
                "FATAL: frozen NVFP4 scalar comparator source drifted"
            )
    return {
        "repo_root": str(REPO_ROOT),
        "repo_git_commit": _repo_commit(),
        "driver_path": str(driver),
        "driver_sha256": file_sha256(driver),
        "isolated_loader_path": str(isolated_loader),
        "isolated_loader_sha256": file_sha256(isolated_loader),
        "numeric_execution_contract_path": str(execution_contract),
        "numeric_execution_contract_sha256": file_sha256(execution_contract),
        "active_corpus_reader_path": str(corpus_reader.resolve(strict=True)),
        "active_corpus_reader_sha256": file_sha256(corpus_reader),
        "frozen_hull": {
            "root": str(LOCKED_HULL_ROOT.resolve(strict=True)),
            "snapshot_tree_sha256": H.snapshot_tree_sha256(),
            "source_sha256": H.source_hashes(),
            "imported_codec_modules": imported_codec_modules,
        },
    }


def _bf16_ladder_module():
    import bf16_ladder as module
    expected = (LOCKED_HULL_ROOT / "bf16_ladder.py").resolve(strict=True)
    actual = Path(module.__file__).resolve(strict=True)
    if actual != expected:
        raise SystemExit("FATAL: bf16_ladder import escaped the locked hull")
    return module


def _corpus_binding(
    args: argparse.Namespace, *, glm_corpus=None,
    manifest_binding: Mapping[str, str] | None = None,
    control_binding: Mapping[str, str] | None = None,
) -> dict[str, object]:
    if args.corpus == "glm":
        if args.glm_manifest is None:
            raise SystemExit("--corpus glm requires --glm-manifest")
        if glm_corpus is None:
            fresh, fresh_binding = load_active_glm_corpus_bound(
                REPO_ROOT, args.glm_manifest
            )
        else:
            if manifest_binding is None:
                raise SystemExit("FATAL: bound GLM manifest identity is missing")
            fresh, fresh_binding = glm_corpus, manifest_binding
        return {
            "manifest_path": str(fresh.manifest_path),
            "manifest_sha256": fresh_binding["sha256"],
            "artifact_path": str(fresh.artifact_path),
            "artifact_sha256": file_sha256(fresh.artifact_path),
            "importance_value_sha256": fresh.manifest["importance_identity"]["value_sha256"],
            "corpus_prismaquant_commit": fresh.manifest["prismaquant_commit"],
        }
    if args.corpus == "dsv4":
        if manifest_binding is None:
            _manifest, manifest_binding = read_bound_json(H.INPUT_MANIFEST)
        if control_binding is None:
            _control, control_binding = read_bound_json(PUBLISHED)
        return {
            "manifest_path": manifest_binding["path"],
            "manifest_sha256": manifest_binding["sha256"],
            "input_path": str(H.INPUT.resolve(strict=True)),
            "input_sha256": file_sha256(H.INPUT),
            "control_path": control_binding["path"],
            "control_sha256": control_binding["sha256"],
        }
    module = _bf16_ladder_module()
    control_present = BF16_PUBLISHED.exists()
    if manifest_binding is None:
        manifest, manifest_binding = read_bound_json(Path(module.MANIFEST))
        loaded_manifest, _names, _entries = module.load_corpus()
        if loaded_manifest != manifest:
            raise SystemExit(
                "FATAL: BF16 manifest changed between bound read and load"
            )
    if control_present and control_binding is None:
        _control, control_binding = read_bound_json(BF16_PUBLISHED)
    return {
        "bf16_ladder_path": str(Path(module.__file__).resolve(strict=True)),
        "bf16_ladder_sha256": file_sha256(Path(module.__file__)),
        "manifest_path": manifest_binding["path"],
        "manifest_sha256": manifest_binding["sha256"],
        "input_path": str(Path(module.INPUT).resolve(strict=True)),
        "input_sha256": file_sha256(Path(module.INPUT)),
        "control_path": str(BF16_PUBLISHED.resolve()),
        "control_present": control_present,
        "control_sha256": (
            control_binding["sha256"] if control_present else None
        ),
    }


def _claim_identity(
    args: argparse.Namespace,
    corpus_binding: Mapping[str, object],
    execution_environment: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "trellis.e2m1_highrate.publication.v1",
        "output": str(args.out.resolve()),
        "corpus": args.corpus,
        "corpus_binding": corpus_binding,
        "glm_rate_plan": args.glm_rate_plan,
        "limit": args.limit,
        "allow_control_drift": bool(args.allow_control_drift),
        "active_sources": _active_source_identity(),
        "environment": dict(execution_environment),
    }


def _verify_final_bindings(
    *,
    args: argparse.Namespace,
    receipt: Mapping[str, object],
    publication_identity: Mapping[str, object],
    execution_environment: Mapping[str, object],
) -> None:
    fresh_environment = _execution_environment()
    if fresh_environment != execution_environment:
        raise SystemExit("FATAL: numeric execution environment drifted during run")
    if receipt.get("environment") != fresh_environment:
        raise SystemExit("FATAL: result execution environment is not the live one")
    if (
        _claim_identity(args, _corpus_binding(args), fresh_environment)
        != publication_identity
    ):
        raise SystemExit("FATAL: publication identity drifted during run")
    if receipt.get("active_source_identity") != _active_source_identity():
        raise SystemExit("FATAL: active/frozen source identity drifted during run")
    binding = receipt.get("corpus_binding")
    if not isinstance(binding, dict):
        raise SystemExit("FATAL: result corpus binding is missing")
    if _corpus_binding(args) != binding:
        raise SystemExit("FATAL: bound corpus/input/control identity drifted during run")


def _prepare_campaign(args: argparse.Namespace) -> dict[str, object]:
    """Load and bind every input before acquiring the output claim."""

    glm_corpus = None
    bf16_ladder = None
    manifest_binding = None
    control_binding = None
    if args.corpus == "dsv4":
        control_document, control_binding = read_bound_json(PUBLISHED)
        manifest_document, manifest_binding = read_bound_json(H.INPUT_MANIFEST)
        published = control_document["per_tensor"]
        entries = {
            str(entry["name"]): entry
            for entry in manifest_document["entries"]
        }
        rate_plan = (*MID_RATES, CONTROL_RATE, *NEW_RATES)
        control_keys = (
            f"tcq_two_tier@{CONTROL_RATE}", "tcq_two_tier@2.5", "tcq_v1@2.5",
        )
    elif args.corpus == "bf16":
        bf16_ladder = _bf16_ladder_module()
        manifest_document, manifest_binding = read_bound_json(
            Path(bf16_ladder.MANIFEST)
        )
        loaded_manifest, _names, entries = bf16_ladder.load_corpus()
        if loaded_manifest != manifest_document:
            raise SystemExit(
                "FATAL: BF16 manifest changed between bound read and load"
            )
        if BF16_PUBLISHED.exists():
            control_document, control_binding = read_bound_json(BF16_PUBLISHED)
            published = control_document["cells"]
        else:
            published = {}
        rate_plan = BF16_RATES
        control_keys = ("tcq_two_tier@2.0",)
    else:
        if args.glm_manifest is None:
            raise SystemExit("--corpus glm requires --glm-manifest")
        glm_corpus, manifest_binding = load_active_glm_corpus_bound(
            REPO_ROOT, args.glm_manifest
        )
        entries = {entry.name: entry for entry in glm_corpus.entries}
        published = {}
        rate_plan = GLM_RATE_PLANS[args.glm_rate_plan]
        control_keys = ()
    names = list(entries)
    if args.limit:
        names = names[:args.limit]
    expected_tensors: dict[str, dict[str, object]] = {}
    expected_controls: dict[str, dict[str, object]] = {}
    for name in names:
        entry = entries[name]
        if args.corpus == "dsv4":
            source_shape = entry["source_weight_shape"]
            importance_shape = entry["importance_shape"]
            logical_shape = [int(source_shape[0]), int(importance_shape[0])]
            population = "dsv4"
        elif args.corpus == "bf16":
            logical_shape = [int(value) for value in entry["source_weight_shape"]]
            population = "bf16"
        else:
            logical_shape = [int(value) for value in entry.source_weight_shape]
            population = entry.population
        expected_tensors[name] = {
            "shape": logical_shape,
            "population": population,
        }
        expected_controls[name] = {}
        for control_key in control_keys:
            published_cell = published.get(name)
            published_arms = (
                published_cell.get("arms", {})
                if isinstance(published_cell, Mapping) else {}
            )
            if control_key not in published_arms:
                raise SystemExit(
                    f"FATAL: {name}: required published control "
                    f"{control_key!r} is absent; refusing before GPU work"
                )
            if (published_cell.get("shape") != logical_shape
                    or published_cell.get("numel") != math.prod(logical_shape)):
                raise SystemExit(
                    f"FATAL: {name}: published control tensor identity "
                    "differs; refusing before GPU work"
                )
            published_arm = published_arms[control_key]
            try:
                validate_e2_published_control_arm(
                    published_arm,
                    key=control_key,
                    shape=logical_shape,
                    weighted_energy=published_cell.get("weighted_energy"),
                    plain_energy=published_cell.get("plain_energy"),
                )
            except CheckpointContractError as exc:
                raise SystemExit(
                    f"FATAL: {name}: invalid published control "
                    f"{control_key!r}: {exc}; refusing before GPU work"
                ) from exc
            expected_controls[name][control_key] = {
                "metrics": {
                    field: published_arm[field]
                    for field in (
                        "weighted_sse", "weighted_nsse", "weighted_snr_db",
                        "plain_sse", "plain_nsse",
                    )
                },
                "footprint": {
                    field: published_arm["footprint"][field]
                    for field in ("total_bytes", "body_rate_q256")
                },
            }
    return {
        "entries": entries,
        "names": names,
        "rate_plan": rate_plan,
        "published": published,
        "control_keys": control_keys,
        "glm_corpus": glm_corpus,
        "bf16_ladder": bf16_ladder,
        "corpus_binding": _corpus_binding(
            args,
            glm_corpus=glm_corpus,
            manifest_binding=manifest_binding,
            control_binding=control_binding,
        ),
        "expected_tensors": expected_tensors,
        "expected_controls": expected_controls,
    }


def main() -> int:
    args = _parse_args()
    if args.corpus != "glm" and args.glm_rate_plan != "scaffold":
        raise SystemExit("--glm-rate-plan is valid only with --corpus glm")
    execution_environment = _execution_environment()
    prepared = _prepare_campaign(args)
    publication_identity = _claim_identity(
        args, prepared["corpus_binding"], execution_environment
    )
    try:
        with exclusive_publication_claim(
            args.out, identity=publication_identity
        ):
            return _run_claimed(
                args, publication_identity, prepared, execution_environment
            )
    except PublicationError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc


def _run_claimed(
    args: argparse.Namespace,
    publication_identity: Mapping[str, object],
    prepared: Mapping[str, object],
    execution_environment: Mapping[str, object],
) -> int:

    partial_path = args.out.with_name(args.out.name + ".partial")
    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"final output already exists (immutable): {args.out}")
    if partial_path.is_symlink():
        raise SystemExit(f"partial output must not be a symlink: {partial_path}")

    if not torch.cuda.is_available():
        raise SystemExit("FATAL: CUDA required (principle 7)")
    device = torch.device("cuda")

    entries = prepared["entries"]
    names = list(prepared["names"])
    rate_plan = prepared["rate_plan"]
    published = prepared["published"]
    control_keys = prepared["control_keys"]
    glm_corpus = prepared["glm_corpus"]
    B = prepared["bf16_ladder"]
    expected_tensors = prepared["expected_tensors"]
    expected_controls = prepared["expected_controls"]

    env = dict(execution_environment)
    active_sources = publication_identity["active_sources"]
    corpus_binding = prepared["corpus_binding"]
    receipt = {
        "schema": "trellis.e2m1_highrate.v3",
        "started_at_unix_s": time.time(),
        "question": ("does the E2M1 trellis, above body rate 2.25 and up to "
                     "its 3.96875 mathematical ceiling, beat scalar NVFP4 "
                     "(4.5 bpw) at equal or smaller bpw on the same corpus"),
        "control_rungs": list(control_keys),
        "arms_measured": ["tcq_two_tier (research 0.28125 plane)",
                          "tcq_v1 (ATTESTED group16_fp8_e4m3_0p5_bpw plane, "
                          "rendered AND priced there)"],
        "control_source": (
            str(PUBLISHED) if args.corpus == "dsv4" else
            (str(BF16_PUBLISHED) if args.corpus == "bf16" else None)
        ),
        "control_rtol": CONTROL_RTOL,
        "corpus": args.corpus,
        "corpus_label": CORPUS_LABEL[args.corpus],
        "corpus_manifest": (
            str(args.glm_manifest.resolve()) if args.corpus == "glm" else None
        ),
        "corpus_binding": corpus_binding,
        "active_source_identity": active_sources,
        "publication_identity_sha256": identity_sha256(publication_identity),
        "glm_rate_plan": args.glm_rate_plan if args.corpus == "glm" else None,
        "aggregation_contract": (
            "dense and routed populations are summarized independently; "
            "no pooled median is valid" if args.corpus == "glm" else
            "single declared corpus population"
        ),
        "rate_plan": list(rate_plan),
        "new_rates": list(NEW_RATES),
        "mathematical_q256_bounds": list(
            TF.FAMILIES[TF.E2M1_FAMILY].mathematical_q256_bounds),
        "arms": ["tcq_two_tier (arm C): trellis rendered on the snapped RESEARCH two-tier plane, priced at both planes", "tcq_v1 (arm D): trellis rendered AND priced on the family's DECLARED group16_fp8_e4m3_0p5_bpw plane -- the arm the NVFP4 comparison needs, since NVFP4 is priced on exactly that plane"],
        "pricing": ("every row carries BOTH prices: exact_bpw at the RESEARCH "
                    "0.28125 bpw two-tier plane (comparable to the published "
                    "ladder) and production_payload_v1.exact_bpw at the "
                    "family's ATTESTED group16_fp8_e4m3_0p5_bpw plane "
                    "(comparable to NVFP4 and to production)"),
        "environment": env,
    }

    resumed = _resume_partial(
        partial_path,
        receipt=receipt,
        expected_tensors=expected_tensors,
        expected_controls=expected_controls,
    )
    if resumed is None:
        resumed_out: dict[str, dict] = {}
    else:
        resumed_out, started_at = resumed
        receipt["started_at_unix_s"] = started_at
        print(
            f"replaying {len(resumed_out)}/{len(names)} checkpoint tensors "
            "before reuse",
            flush=True,
        )
    out: dict[str, dict] = {}
    for index, name in enumerate(names, start=1):
        saved_cell = resumed_out.get(name)
        entry = entries[name]
        if args.corpus == "dsv4":
            packed, raw_scale, importance = W.load_compact(name)
            for label, value, key in (
                    ("weight", packed, "source_weight_sha256"),
                    ("scale", raw_scale, "source_scale_sha256"),
                    ("importance", importance, "importance_sha256")):
                if H.tensor_sha256(value) != entry[key]:
                    raise SystemExit(f"{name}: compact {label} hash mismatch")
            weight = W.dequant_mxfp4(packed, raw_scale, device)
            importance = importance.to(device, torch.float32)
        elif args.corpus == "bf16":
            assert B is not None
            raw, imp = B.load_tensor(entry)   # hashes checked inside
            weight = raw.to(device, torch.float32)
            importance = imp.to(device, torch.float32)
        else:
            assert glm_corpus is not None
            raw, imp = glm_corpus.load_tensor(entry)
            weight = raw.to(device, torch.float32)
            importance = imp.to(device, torch.float32)
        rows, columns = map(int, weight.shape)
        eff = P.eff_scale_plane(weight)
        H.assert_context_parity(weight, importance, eff)
        _, _, metric_w, _ = H.context_from_plane(weight, importance, eff)
        weighted_energy = C.weighted_sse(weight, torch.zeros_like(weight),
                                         metric_w)
        plain_energy = C.plain_sse(weight, torch.zeros_like(weight))

        # Arm D context: rendered AND priced on the family's DECLARED
        # group16_fp8_e4m3_0p5_bpw plane.  This is the arm that answers the
        # NVFP4 question without any plane substitution -- NVFP4 is priced on
        # exactly this plane -- so it is measured here alongside arm C rather
        # than left to a caveat.
        x_v1, pes_v1, _, enc_v1 = H.context_from_plane(weight, importance, eff)
        codes_v1, _, alpha_v1 = W.alphabets(x_v1, enc_v1)
        colw_v1 = S4.column_weight(enc_v1)

        snapped, _, _, snap_stats = H.two_tier_snap_plane(eff)
        x_tt, pes_tt, _, enc_tt = H.context_from_plane(weight, importance,
                                                       snapped)
        codes_tt, _, alpha_tt = W.alphabets(x_tt, enc_tt)
        colw_tt = S4.column_weight(enc_tt)

        cell = {"shape": [rows, columns], "numel": rows * columns,
                "population": (
                    entry.population if args.corpus == "glm" else args.corpus
                ),
                "weighted_energy": weighted_energy,
                "plain_energy": plain_energy,
                "two_tier_plane_sha256": H.tensor_sha256(snapped),
                "arms": {}}

        # scalar NVFP4 RTN on this tensor, for the per-subset contrast: a
        # bypass column IS this, so "trellis on bypass columns" and "NVFP4 on
        # bypass columns" must agree up to the scale plane, while a genuine
        # coding gain can only live on the SHAPED columns.
        nvfp4_recon = NVFP4_COMPARATOR._rtn_dequant_nvfp4(
            weight, group_size=16
        ).to(weight.dtype)

        def scalar_subgrid(x_cols, enc_cols, pes_cols, w_cols, m_levels):
            """RTN onto the weighted-MSE-optimal m-level subset of the E2M1
            grid -- the honest same-rate, same-grid, same-plane partner for a
            shaped trellis column.

            `P.best_level_subset` is EXHAUSTIVE (C(15,2)=105, C(15,4)=1365,
            C(15,8)=6435) and is the SAME routine the trellis uses to choose
            its OWN alphabet, so the two sides differ in the coder and nothing
            else.  A rate-R trellis step gets 2^(R+1) levels and picks among
            2^R of them per state; the scalar partner gets a free choice among
            2^R levels.  Equal bits, equal grid, equal plane.
            """
            levels, _ = P.best_level_subset(x_cols, enc_cols, m_levels)
            recon_s = P.quantize_to_levels(x_cols, levels, pes_cols)
            return levels, recon_s

        # The "shared" scope fits the subset on the WHOLE tensor, so it is
        # identical across every column class and every rung of one lane.
        # Memoized: on the bf16 corpus (25M elements/tensor) recomputing it
        # per class per rung is most of the run.
        shared_memo: dict = {}

        def subset_split(recon, rate, colw_lane, x_lane, enc_lane, pes_lane,
                         lane_tag):
            """Per-schedule-rate weighted SNR, trellis vs scalar NVFP4, over
            the SAME columns.  Answers whether the shaped columns carry gain
            the bypass columns cannot."""
            try:
                sched, _ = S4.build_schedules(rate, columns, colw_lane,
                                              include_variants=False)
            except AssertionError:
                return None
            sched = torch.as_tensor(np.asarray(sched["rwf"]),
                                    device=weight.device)
            out = {}
            for r in (1, 2, 3, 4):
                mask = sched == r
                n = int(mask.sum())
                if not n:
                    continue
                cols_idx = mask.nonzero(as_tuple=True)[0]
                w_s = weight.index_select(1, cols_idx)
                m_s = metric_w.index_select(1, cols_idx)
                e = C.weighted_sse(w_s, torch.zeros_like(w_s), m_s)
                t = C.weighted_sse(w_s, recon.index_select(1, cols_idx)
                                   .to(weight.dtype), m_s)
                v = C.weighted_sse(w_s, nvfp4_recon.index_select(1, cols_idx)
                                   .to(weight.dtype), m_s)
                row = {
                    "columns": n, "energy": e,
                    "trellis_wsse": t, "nvfp4_wsse": v,
                    "trellis_db": 10.0 * math.log10(e / t),
                    "nvfp4_db": 10.0 * math.log10(e / v),
                    "trellis_minus_nvfp4_db": 10.0 * math.log10(v / t),
                    "bits_per_weight_here": r,
                    "nvfp4_bits_per_weight": 4,
                }
                # --- the same-grid scalar control, the real coding gain ----
                x_s = x_lane.index_select(1, cols_idx)
                enc_s = enc_lane.index_select(1, cols_idx)
                pes_s = pes_lane.index_select(1, cols_idx)
                # rate 4 is bypass: no alphabet at all, the whole grid.
                m_levels = (1 << r) if r < 4 else len(P.E2M1_LEVELS)
                for scope in ("oracle", "shared"):
                    if scope == "oracle":
                        levels, _ = scalar_subgrid(
                            x_s, enc_s, pes_s, w_s, m_levels)
                    else:
                        key = (lane_tag, m_levels)
                        if key not in shared_memo:
                            shared_memo[key] = P.best_level_subset(
                                x_lane, enc_lane, m_levels)[0]
                        levels = shared_memo[key]
                    recon_s = P.quantize_to_levels(x_s, levels, pes_s)
                    q = C.weighted_sse(w_s, recon_s.to(weight.dtype), m_s)
                    row[f"scalar_subgrid_{scope}"] = {
                        "levels": [float(v) for v in levels],
                        "n_levels": m_levels,
                        "wsse": q,
                        "db": 10.0 * math.log10(e / q),
                        "coding_gain_db": 10.0 * math.log10(q / t),
                        "subset_fit_scope": (
                            "these columns only (ORACLE: a tougher baseline "
                            "than the trellis gets, whose alphabet is fit "
                            "tensor-wide -- so a positive gain here is "
                            "conservative)" if scope == "oracle" else
                            "whole tensor (the SAME fitting scope the trellis "
                            "alphabet gets)"),
                    }
                out[str(r)] = row
            return out

        def emit(key, arm, rung, recon, seconds, footprint, extra=None,
                 _cell=cell):
            row = W.metric_row(weight, recon, metric_w, weighted_energy,
                               plain_energy, seconds, footprint)
            row.update({"arm": arm, "rung": rung})
            if extra:
                row.update(dict(extra))
            row["subset_split"] = subset_split(
                recon, rung,
                *( (colw_v1, x_v1, enc_v1, pes_v1, "v1") if arm == "tcq_v1"
                   else (colw_tt, x_tt, enc_tt, pes_tt, "two_tier") ))
            _cell["arms"][key] = row

        unreachable = []
        for lane, plane_args, coding, colw in (
                ("tcq_two_tier", (x_tt, enc_tt, pes_tt, codes_tt),
                 H.SCALE_CODING_TWO_TIER, colw_tt),
                ("tcq_v1", (x_v1, enc_v1, pes_v1, codes_v1),
                 H.SCALE_CODING_V1, colw_v1)):
            x, enc, pes, codes = plane_args
            for rate in rate_plan:
                try:
                    H.emit_trellis(name, lane, rate, weight, x, enc, pes,
                                   codes, colw, coding, "triton", emit)
                except AssertionError as exc:
                    unreachable.append({"lane": lane, "rate": rate,
                                        "reason": str(exc)})
                    print(f"      {lane}@{rate}: UNREACHABLE ({exc})",
                          flush=True)
        cell["unreachable_rungs"] = unreachable

        # --- the control -------------------------------------------------
        checks = {}
        footprint_equal = True
        for control_key in control_keys:
            if (name not in published
                    or control_key not in published[name].get("arms", {})
                    or control_key not in cell["arms"]):
                checks[f"{control_key}.MISSING"] = {
                    "mine": None, "published": None, "rel": 0.0}
                continue
            mine = cell["arms"][control_key]
            theirs = published[name]["arms"][control_key]
            # ``bf16_w4a4_results.json`` predates ``plain_snr_db``.  The
            # published row does carry the primitive plain-domain quantities
            # (SSE and normalized SSE), so compare those directly instead of
            # requiring a derived field which cannot add information.  Keep
            # the weighted SNR because that is the study's decision metric.
            for field in ("weighted_sse", "weighted_nsse", "weighted_snr_db",
                          "plain_sse", "plain_nsse"):
                a, b = float(mine[field]), float(theirs[field])
                checks[f"{control_key}.{field}"] = {
                    "mine": a, "published": b,
                    "rel": abs(a - b) / max(abs(b), 1e-300)}
            footprint_equal = footprint_equal and (
                mine["footprint"]["total_bytes"]
                == theirs["footprint"]["total_bytes"]
                and mine["footprint"]["body_rate_q256"]
                == theirs["footprint"]["body_rate_q256"])
        measured = [c["rel"] for c in checks.values()
                    if c["mine"] is not None]
        worst = max(measured) if measured else None
        if not measured:
            footprint_equal = False
        status = ("pass" if (measured and worst <= CONTROL_RTOL
                             and footprint_equal)
                  else ("uncontrolled" if not measured else "fail"))
        cell["control"] = {"status": status, "worst_relative": worst,
                           "footprint_equal": footprint_equal,
                           "checks": checks}
        worst_text = f"{worst:.3e}" if worst is not None else "unmeasured"
        print(f"[{index}/{len(names)}] {name}: control {status.upper()} "
              f"(worst rel {worst_text})", flush=True)
        for rate in rate_plan:
            c_row = cell["arms"].get(f"tcq_two_tier@{rate}")
            d_row = cell["arms"].get(f"tcq_v1@{rate}")
            if c_row is None or d_row is None:
                print(f"      R={rate:<8} (unreachable on one or both arms)",
                      flush=True)
                continue
            print(f"      R={rate:<8} armC {c_row['weighted_snr_db']:7.3f} dB "
                  f"@{c_row['footprint']['exact_bpw']:.4f} research / "
                  f"{c_row['footprint']['production_payload_v1']['exact_bpw']:.4f} "
                  f"attested | armD {d_row['weighted_snr_db']:7.3f} dB "
                  f"@{d_row['footprint']['exact_bpw']:.4f} attested",
                  flush=True)
        if status == "fail" and not args.allow_control_drift:
            raise SystemExit(
                f"FATAL: {name}: a control rung does not reproduce the "
                f"published row (worst relative {worst_text}, footprint_equal "
                f"{footprint_equal}). These rows are NOT comparable to the "
                f"published ladder; re-run under the pinned environment "
                f"(hull_sweep.py --print-container-command).")
        if saved_cell is not None:
            _require_e2_replay_match(name, saved_cell, cell)
            print(
                f"[{index}/{len(names)}] {name}: REPLAY VERIFIED",
                flush=True,
            )
        out[name] = cell
        checkpoint = _checkpoint_document(receipt, out, partial=True)
        try:
            validate_e2m1_checkpoint(
                checkpoint, current_receipt=receipt,
                expected_tensors=expected_tensors,
                expected_controls=expected_controls,
            )
        except CheckpointContractError as exc:
            raise SystemExit(
                f"FATAL: refusing invalid generated checkpoint: {exc}"
            ) from exc
        _atomic_json(
            partial_path,
            checkpoint,
        )

    receipt["completed_at_unix_s"] = time.time()
    receipt["tensors_done"] = len(out)
    receipt["status"] = "ok"
    receipt["control_verdict"] = {
        n: c["control"]["status"] for n, c in out.items()}
    receipt["population_counts"] = {
        population: sum(cell["population"] == population for cell in out.values())
        for population in sorted({cell["population"] for cell in out.values()})
    }
    final_checkpoint = _checkpoint_document(receipt, out, partial=False)
    try:
        validate_e2m1_checkpoint(
            final_checkpoint, current_receipt=receipt,
            expected_tensors=expected_tensors,
            expected_controls=expected_controls,
            require_partial=False,
        )
    except CheckpointContractError as exc:
        raise SystemExit(f"FATAL: refusing invalid final result: {exc}") from exc
    _atomic_json(partial_path, final_checkpoint)
    _verify_final_bindings(
        args=args,
        receipt=receipt,
        publication_identity=publication_identity,
        execution_environment=execution_environment,
    )
    try:
        publish_file_no_replace(partial_path, args.out)
    except PublicationError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
