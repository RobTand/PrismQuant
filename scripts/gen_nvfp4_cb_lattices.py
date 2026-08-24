"""Generate the committed NVFP4-CB universal lattice cache.

Builds the canonical fixed codebooks serialized into artifact sidecars.
The public product domain is derived from ``prismaquant.cb_layout``. Explicit
width-14..16 fp4/d4 research tables remain materialized for direct-codec and
kernel study, but they do not create public K26..K32 format ids.

Merge semantics: existing tables in data/nvfp4_cb_lattices.pt are PRESERVED
and only missing keys are built.  Missing canonical tables are built explicitly
on CPU so the generator cannot silently inherit the machine's CUDA topology.
The resulting asset still becomes canonical only after review and updating its
digest pin. Missing fp4/d4 widths 0..5 and 13..16 are exactly reconstructible
from the versioned nested definition. Every other cache-miss builder remains
research-only. Delete the .pt first to force a full rebuild.

    PYTHONPATH=. python scripts/gen_nvfp4_cb_lattices.py
"""
from __future__ import annotations

import hashlib

import torch

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_layout import (
    FAMILIES,
    VEC_DIM,
    codebook_subtable_shapes,
    subtable_bit_widths,
)


def required_lattice_specs() -> tuple[tuple[int, str, int, bool], ...]:
    """Unique public lattice tables plus explicit low-level research widths."""

    wanted: list[tuple[int, str, int, bool]] = []

    def add(item: tuple[int, str, int, bool]) -> None:
        if item not in wanted:
            wanted.append(item)

    # Full mode is an intentionally separate research surface.
    for k in (12, 13, 14):
        add((k, "fp4", VEC_DIM, False))

    # Direct-codec/kernel research reaches the uint32 endpoint using two d4
    # width-16 tables. These assets are intentionally not derived from a
    # public format rung: NVFP4_CB_K26..K32 do not exist in the registry.
    for width in (14, 15, 16):
        add((width, "fp4", VEC_DIM // 2, False))

    for family in FAMILIES:
        # The positive-magnitude (half-grid) lattice existed only for the
        # signed family, deleted 2026-08-17; every surviving family codes a
        # sign-symmetric table. Fail closed rather than let a future mode
        # silently inherit `positive=False` and get the wrong lattice.
        if family.mode not in ("product", "full"):
            raise ValueError(
                f"{family.prefix}: unhandled CB mode {family.mode!r} -- declare "
                "whether its lattice is positive-magnitude before generating")
        positive = False
        for k in family.accepted_rungs:
            widths = subtable_bit_widths(k, family.mode, family.n_sub)
            shapes = codebook_subtable_shapes(
                k, family.mode, family.n_sub
            )
            for width, (_, dimension) in zip(widths, shapes):
                add((width, family.grid, dimension, positive))
    return tuple(wanted)


def main() -> None:
    out: dict[str, torch.Tensor] = {}
    if cb._DATA.exists():
        out.update(torch.load(cb._DATA, map_location="cpu",
                              weights_only=True))

    built = 0
    for k, grid, d, positive in required_lattice_specs():
        key = cb._lattice_key(k, grid, d, positive)
        structured = cb._is_structured_fp4_d4_key(k, grid, d, positive)
        if key in out and not structured:
            print(f"kept  {key}: {tuple(out[key].shape)}")
            continue
        # fp4/d4 widths 0..5 and 13..16 use the exact nested construction;
        # other research/FP8 tables retain the historical seeded CPU Lloyd
        # path. Recompute structured keys cheaply so the asset cannot retain a
        # stale construction-version table after this script is reviewed.
        generated = cb._build_lattice(
            k,
            grid,
            d,
            positive=positive,
            device="cpu",
        )
        if key in out and torch.equal(out[key], generated):
            print(f"kept  {key}: {tuple(out[key].shape)}")
            continue
        action = "rebuilt" if key in out else "built"
        out[key] = generated
        built += 1
        print(f"{action} {key}: {tuple(out[key].shape)}")
    cb._DATA.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cb._DATA)
    print(f"wrote {cb._DATA} ({len(out)} tables, {built} new)")
    print(
        "asset sha256 (copy to _LATTICE_ASSET_SHA256 after review): "
        f"{hashlib.sha256(cb._DATA.read_bytes()).hexdigest()}"
    )
    print(
        "structured fp4/d4 construction: "
        f"{cb.STRUCTURED_FP4_D4_LATTICE_VERSION}"
    )


if __name__ == "__main__":
    main()
