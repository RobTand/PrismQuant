"""Deterministic, serving-safe plans for parallel production-cache fills.

The renderer is embarrassingly parallel across recipe qnames, but splitting a
fused serving unit or scattering every decoder layer across every worker makes
the later union harder to audit and defeats layer-local source prefetch.  This
module bins whole decoder layers (plus indivisible auxiliary groups) with a
simple longest-processing-time heuristic derived from the probe shapes.

The output is only a plan.  Workers still write independent
``ProductionWeightCache`` manifests, and ``union_production_cache`` owns the
fail-closed reconciliation of those manifests and their backing tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from prismaquant.decision_units import block_id_from_qname
from prismaquant.model_profiles import detect_profile


SCHEMA = "prismaquant.production_cache_stripe_plan.v1"
_LAYER_RE = re.compile(r"(?:^|[.])layers[.](\d+)(?:[.]|$)")


@dataclass(frozen=True)
class Stripe:
    index: int
    qnames: tuple[str, ...]
    groups: tuple[str, ...]
    estimated_work: int
    parameters: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_key(qname: str) -> str:
    if qname.startswith("mtp."):
        return "aux:mtp"
    if qname == "lm_head" or qname.endswith(".lm_head"):
        return "aux:lm_head"
    block = block_id_from_qname(qname)
    if _LAYER_RE.search(block):
        return f"layer:{block}"
    return f"unit:{qname}"


def _entry_work(entry: Mapping[str, object]) -> int:
    """Shape-weighted render work proxy used only for load balancing.

    GPTQ's dominant covariance/solve work grows with the input width as well
    as parameter count.  The product is deliberately integer and hardware
    independent; observed pilot timings can replace it in a future schema.
    """
    n_params = int(entry.get("n_params", 0) or 0)
    in_features = int(entry.get("in_features", 0) or 0)
    return n_params * max(in_features, 1)


def plan_stripes(
    stats: Mapping[str, Mapping[str, object]],
    *,
    profile,
    n_stripes: int,
) -> tuple[Stripe, ...]:
    if n_stripes < 1:
        raise ValueError("n_stripes must be positive")
    names = tuple(sorted(
        str(name) for name, entry in stats.items()
        if isinstance(entry, Mapping) and int(entry.get("n_params", 0) or 0) > 0
    ))
    if not names:
        raise ValueError("probe stats contain no positive-size qnames")

    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(_group_key(name), []).append(name)

    # A whole-layer grouping is stricter than the serving-unit constraint, but
    # assert the latter explicitly so a future auxiliary grouping cannot split
    # a fused/packed unit by accident.
    qname_group = {
        qname: group for group, members in groups.items() for qname in members
    }
    for accessor in ("fused_sibling_group", "packed_expert_format_group"):
        method = getattr(profile, accessor, None)
        if not callable(method):
            continue
        serving_groups: dict[str, set[str]] = {}
        for name in names:
            key = method(name)
            if key is not None:
                serving_groups.setdefault(str(key), set()).add(name)
        for key, members in serving_groups.items():
            owners = {qname_group[name] for name in members}
            if len(owners) != 1:
                raise RuntimeError(
                    f"serving unit {accessor}:{key} spans stripe groups: "
                    f"{sorted(owners)}"
                )

    weighted_groups: list[tuple[int, int, str, tuple[str, ...]]] = []
    for group, members_raw in groups.items():
        members = tuple(sorted(members_raw))
        params = sum(int(stats[name].get("n_params", 0) or 0) for name in members)
        work = sum(_entry_work(stats[name]) for name in members)
        weighted_groups.append((work, params, group, members))
    weighted_groups.sort(key=lambda row: (-row[0], -row[1], row[2]))

    bins: list[dict[str, object]] = [
        {"work": 0, "params": 0, "groups": [], "qnames": []}
        for _ in range(n_stripes)
    ]
    for work, params, group, members in weighted_groups:
        index = min(
            range(n_stripes),
            key=lambda idx: (int(bins[idx]["work"]), idx),
        )
        target = bins[index]
        target["work"] = int(target["work"]) + work
        target["params"] = int(target["params"]) + params
        target["groups"].append(group)  # type: ignore[union-attr]
        target["qnames"].extend(members)  # type: ignore[union-attr]

    stripes = tuple(
        Stripe(
            index=index,
            qnames=tuple(sorted(bins[index]["qnames"])),  # type: ignore[arg-type]
            groups=tuple(sorted(bins[index]["groups"])),  # type: ignore[arg-type]
            estimated_work=int(bins[index]["work"]),
            parameters=int(bins[index]["params"]),
        )
        for index in range(n_stripes)
    )
    flattened = [name for stripe in stripes for name in stripe.qnames]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(names):
        raise RuntimeError("stripe plan is not an exact disjoint qname cover")
    return stripes


def _load_probe(path: Path) -> Mapping[str, Mapping[str, object]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # trusted local pipeline artifact
    stats = payload.get("stats") if isinstance(payload, Mapping) else None
    if not isinstance(stats, Mapping):
        raise ValueError(f"probe has no stats mapping: {path}")
    return stats


def write_plan(
    *,
    probe_path: Path,
    model_path: str,
    output_dir: Path,
    n_stripes: int,
    formats: Sequence[str],
) -> Path:
    profile = detect_profile(model_path)
    stripes = plan_stripes(
        _load_probe(probe_path), profile=profile, n_stripes=n_stripes
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    for stripe in stripes:
        path = output_dir / f"stripe-{stripe.index:02d}.qnames.txt"
        path.write_text("".join(f"{name}\n" for name in stripe.qnames), encoding="utf-8")
        files.append({
            "index": stripe.index,
            "path": path.name,
            "sha256": _sha256(path),
            "qnames": len(stripe.qnames),
            "groups": list(stripe.groups),
            "estimated_work": stripe.estimated_work,
            "parameters": stripe.parameters,
        })
    manifest = {
        "schema": SCHEMA,
        "probe": str(probe_path.resolve()),
        "probe_sha256": _sha256(probe_path),
        "model": str(Path(model_path).resolve()),
        "profile": profile.name,
        "formats": list(formats),
        "n_stripes": n_stripes,
        "qnames": sum(int(item["qnames"]) for item in files),
        "stripes": files,
    }
    manifest_path = output_dir / "stripe-plan.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan disjoint, layer-local ProductionWeightCache stripes"
    )
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stripes", type=int, default=2)
    parser.add_argument("--formats", default="NVFP4,FP8_E4M3")
    args = parser.parse_args(argv)
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    if not formats:
        parser.error("--formats must name at least one format")
    path = write_plan(
        probe_path=args.probe,
        model_path=args.model,
        output_dir=args.output_dir,
        n_stripes=args.stripes,
        formats=formats,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
