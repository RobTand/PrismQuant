#!/usr/bin/env python3
"""Cross-box determinism preflight for a unit-sharded distributed render.

A distributed render is only correct if both boxes turn the same source
weight into the same bytes. Same GB10, same wheel, same levers — expected
bit-identical; but "expected" is not measured, and a render that differs
across boxes turns a merged cache into two half-artifacts wearing one name.
So the campaign runs this first, on both boxes, and refuses to launch unless
the manifests match exactly.

    # on each box
    python tools/cluster_render_sentinel.py render \\
        --model /home/rob/models/Qwen3-0.6B \\
        --k 8 --formats NVFP4,FP8_DYNAMIC \\
        --output /work/sentinel-$(hostname).json

    # on the coordinator
    python tools/cluster_render_sentinel.py compare \\
        --manifest /work/sentinel-boxA.json \\
        --manifest /work/sentinel-boxB.json

``compare`` exits non-zero on ANY mismatch and names the first differing
units. The unit pick is deterministic (evenly spaced over the model's own
enumeration order), so both boxes render the same K units without
coordination.

Determinism note: ``PRISMAQUANT_DETERMINISTIC=1`` is what makes the GPTQ
Cholesky/U-update reduction order reproducible (see
``build_production_cache.py``). The sentinel records whether it was set and
``compare`` refuses to certify two manifests that disagree about it — a
cross-box match under nondeterministic reductions is luck, and a mismatch
under them is uninformative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

SCHEMA = "prismaquant.cluster_render_sentinel.v1"


def select_sentinel_units(qnames, k: int) -> list[str]:
    """Deterministically pick K units, evenly spaced over the enumeration.

    Even spacing (rather than the first K) makes the pick span the depth of
    the model, so a divergence confined to late layers is still caught.
    """
    names = [str(name) for name in qnames]
    if k < 1:
        raise ValueError("k must be >= 1")
    if k >= len(names):
        return names
    step = len(names) / float(k)
    picked: list[str] = []
    for i in range(k):
        index = int(i * step)
        if names[index] not in picked:
            picked.append(names[index])
    return picked


def _tensor_digest(tensor: torch.Tensor) -> dict:
    contiguous = tensor.detach().cpu().contiguous()
    payload = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _render(args) -> int:
    from prismaquant.build_rtn_cache import iter_quantizable_tensors
    from prismaquant.calibration_data import (
        _dtype_from_name,
        load_wikitext_calibration_windowed,
    )
    from prismaquant.gpu_guard import require_cuda_hot_path
    from prismaquant.model_profiles import detect_profile_with_warning
    from prismaquant.perturbed_x_cache import calibration_data_hash
    from prismaquant.production_weight_cache import (
        fill_production_weight_cache,
    )
    from prismaquant.unit_sharding import host_identity
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = require_cuda_hot_path("cluster_render_sentinel")
    formats = [f.strip().upper() for f in args.formats.split(",") if f.strip()]
    levers = {
        name: True
        for name in (x.strip() for x in args.enable.split(","))
        if name
    }
    dtype = _dtype_from_name(args.dtype)

    local_only = Path(args.model).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=local_only
    )
    calib_ids = load_wikitext_calibration_windowed(
        tokenizer,
        args.n_calib_samples,
        args.calib_seqlen,
        split=args.calib_split,
        seed=args.calib_seed,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=local_only,
        device_map="cuda" if device.type == "cuda" else None,
    )
    model.eval()
    profile = detect_profile_with_warning(
        args.model, entrypoint="cluster-render-sentinel"
    )
    skip_tokens = list(profile.pinned_names())
    qnames: list[str] = []
    for full_name, mod, attr in iter_quantizable_tensors(model, profile):
        if attr != "weight" or not isinstance(mod, nn.Linear):
            continue
        qname = (
            full_name[:-7] if full_name.endswith(".weight") else full_name
        )
        if any(s in qname.split(".") for s in skip_tokens):
            continue
        qnames.append(qname)

    picked = select_sentinel_units(qnames, args.k)
    print(
        f"[sentinel] rendering {len(picked)} of {len(qnames)} units x "
        f"{formats}",
        flush=True,
    )
    # Pass the FULL enumeration and render only the picked units. The
    # activation reservoir is shared across hooked Linears, so a run that
    # hooked only K units would sample different rows and the sentinel would
    # be certifying bytes the real build never produces. `render_qnames`
    # narrows the render and nothing else.
    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames,
        formats=formats,
        levers=levers,
        max_act_rows=args.max_act_rows,
        progress=False,
        render_qnames=picked,
    )
    entries = {
        f"{qname}|{fmt}": _tensor_digest(cache.get(qname, fmt))
        for (qname, fmt) in sorted(cache.weights)
    }
    # Fail closed on a partial render. The cache fill can drop a (unit,
    # format) it could not render (2026-08-23: sparklina lacked python3-dev,
    # Triton could not build, every NVFP4 entry was silently absent while
    # the FP8 entries matched); a manifest that omits them would certify a
    # box that cannot render the format at all.
    missing = missing_entries({"units": picked, "formats": formats,
                               "entries": entries})
    if missing:
        print(
            "[sentinel] ERROR: incomplete render — refusing to write a "
            f"manifest. {len(missing)} of {len(picked) * len(formats)} "
            f"(unit, format) pairs did not render on this box: "
            f"{missing[:8]}. Look for build/toolchain errors above.",
            flush=True,
        )
        return 2
    records = (cache.metadata or {}).get("render_scores") or {}
    payload = {
        "schema": SCHEMA,
        "model": str(args.model),
        "formats": formats,
        "levers": sorted(k for k, v in levers.items() if v),
        "k": int(args.k),
        "units": picked,
        "calibration": {
            "n_calib_samples": int(args.n_calib_samples),
            "calib_seqlen": int(args.calib_seqlen),
            "calib_split": str(args.calib_split),
            "calib_seed": int(args.calib_seed),
            "max_act_rows": int(args.max_act_rows),
            "calib_hash": calibration_data_hash(calib_ids),
        },
        "deterministic_mode": os.environ.get(
            "PRISMAQUANT_DETERMINISTIC", "0"
        ),
        "host": host_identity(),
        "entries": entries,
        "render_scores": dict(sorted((records.get("records") or {}).items())),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(
        f"[sentinel] wrote {output} entries={len(entries)} "
        f"digest={_manifest_digest(payload)[:16]}",
        flush=True,
    )
    return 0


def _manifest_digest(payload) -> str:
    comparable = {
        key: payload[key]
        for key in ("schema", "formats", "levers", "units", "entries")
    }
    return hashlib.sha256(
        json.dumps(comparable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def missing_entries(manifest) -> list[str]:
    """(unit, format) keys a manifest promises but did not render.

    Entries are keyed by the cache's canonical format name (the CLI alias
    FP8_DYNAMIC renders as FP8_E4M3), so requested formats are canonicalized
    before they are looked up.
    """
    try:
        from prismaquant.format_registry import canonical_format_name
    except ImportError:  # compare-only hosts without the package
        def canonical_format_name(name):
            return name
    expected = {
        f"{unit}|{canonical_format_name(fmt)}"
        for unit in manifest.get("units") or []
        for fmt in manifest.get("formats") or []
    }
    return sorted(expected - set(manifest.get("entries") or {}))


def compare_manifests(manifests) -> list[str]:
    """Return the list of problems; empty means the boxes agree."""
    problems: list[str] = []
    for index, manifest in enumerate(manifests):
        missing = missing_entries(manifest)
        if missing:
            host = (manifest.get("host") or {}).get("hostname", f"#{index}")
            problems.append(
                f"manifest {host} is incomplete: {len(missing)} of its "
                f"promised (unit, format) pairs are absent: {missing[:8]}"
            )
    first = manifests[0]
    for other in manifests[1:]:
        for field in ("schema", "formats", "levers", "units", "k"):
            if first.get(field) != other.get(field):
                problems.append(
                    f"{field} differs: {first.get(field)!r} vs "
                    f"{other.get(field)!r}"
                )
        if first.get("calibration") != other.get("calibration"):
            problems.append(
                "calibration contract differs: "
                f"{first.get('calibration')!r} vs {other.get('calibration')!r}"
            )
        if str(first.get("deterministic_mode")) != str(
            other.get("deterministic_mode")
        ):
            problems.append(
                "PRISMAQUANT_DETERMINISTIC differs across boxes: "
                f"{first.get('deterministic_mode')!r} vs "
                f"{other.get('deterministic_mode')!r}"
            )
        a_entries = first.get("entries") or {}
        b_entries = other.get("entries") or {}
        only_a = sorted(set(a_entries) - set(b_entries))
        only_b = sorted(set(b_entries) - set(a_entries))
        if only_a:
            problems.append(f"entries only in the first manifest: {only_a[:8]}")
        if only_b:
            problems.append(f"entries only in a later manifest: {only_b[:8]}")
        differing = sorted(
            key
            for key in set(a_entries) & set(b_entries)
            if a_entries[key] != b_entries[key]
        )
        if differing:
            sample = ", ".join(
                f"{key} {a_entries[key].get('sha256', '')[:12]} != "
                f"{b_entries[key].get('sha256', '')[:12]}"
                for key in differing[:5]
            )
            problems.append(
                f"{len(differing)} of {len(a_entries)} rendered units differ: "
                f"{sample}"
            )
    return problems


def _compare(args) -> int:
    manifests = [json.loads(Path(p).read_text()) for p in args.manifest]
    if len(manifests) < 2:
        raise SystemExit(
            "[sentinel] ERROR: --manifest must be given at least twice."
        )
    problems = compare_manifests(manifests)
    if problems:
        print("[sentinel] CROSS-BOX RENDER MISMATCH — refuse the run:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    if str(manifests[0].get("deterministic_mode", "0")).lower() not in {
        "1",
        "true",
        "yes",
    }:
        print(
            "[sentinel] CROSS-BOX RENDER MISMATCH — refuse the run:\n"
            "  - manifests match but were rendered with "
            "PRISMAQUANT_DETERMINISTIC unset; the GPTQ reduction order is "
            "not reproducible, so the match does not certify anything. "
            "Re-run both boxes with PRISMAQUANT_DETERMINISTIC=1."
        )
        return 1
    hosts = [m.get("host", {}).get("hostname") for m in manifests]
    print(
        f"[sentinel] OK: {len(manifests)} manifests agree on "
        f"{len(manifests[0].get('entries') or {})} rendered units "
        f"(hosts={hosts}, "
        f"digest={_manifest_digest(manifests[0])[:16]})"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="Render K fixed units, emit a manifest.")
    render.add_argument("--model", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--k", type=int, default=8)
    render.add_argument("--formats", default="NVFP4,FP8_DYNAMIC")
    render.add_argument(
        "--enable", default="gptq,static_act_order,joint_scale_opt"
    )
    render.add_argument("--n-calib-samples", type=int, default=8)
    render.add_argument("--calib-seqlen", type=int, default=256)
    render.add_argument("--calib-split", default="train")
    render.add_argument("--calib-seed", type=int, default=42)
    render.add_argument("--max-act-rows", type=int, default=512)
    render.add_argument("--dtype", default="bf16")
    render.set_defaults(func=_render)

    compare = sub.add_parser("compare", help="Diff two or more manifests.")
    compare.add_argument("--manifest", action="append", required=True)
    compare.set_defaults(func=_compare)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
