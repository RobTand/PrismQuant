"""Generate the committed NVFP4-CB universal lattice cache.

Builds the grid-snapped Lloyd lattices used as the fixed (no-sidecar)
codebooks. Production product coverage is derived from
``prismaquant.cb_layout``; only the explicit full-mode research tables are
listed locally.

Merge semantics: existing tables in data/nvfp4_cb_lattices.pt are PRESERVED
and only missing keys are built.  Missing canonical tables are built explicitly
on CPU so the generator cannot silently inherit the machine's CUDA topology.
The resulting asset still becomes canonical only after review and updating its
digest pin; the runtime's cache-miss builder remains research-only.  Delete the
.pt first to force a full rebuild.

    PYTHONPATH=. python scripts/gen_nvfp4_cb_lattices.py
"""
from __future__ import annotations

import torch

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_layout import (
    FAMILIES,
    VEC_DIM,
    codebook_subtable_shapes,
    subtable_bit_widths,
)


def required_lattice_specs() -> tuple[tuple[int, str, int, bool], ...]:
    """Unique lattice tables required by every declared producer family."""

    wanted: list[tuple[int, str, int, bool]] = []

    def add(item: tuple[int, str, int, bool]) -> None:
        if item not in wanted:
            wanted.append(item)

    # Full mode is an intentionally separate research surface.
    for k in (12, 13, 14):
        add((k, "fp4", VEC_DIM, False))

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
        for k in family.rungs:
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
        if key in out:
            print(f"kept  {key}: {tuple(out[key].shape)}")
            continue
        out[key] = cb._build_lattice(
            k,
            grid,
            d,
            positive=positive,
            device="cpu",
        )
        built += 1
        print(f"built {key}: {tuple(out[key].shape)}")
    cb._DATA.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cb._DATA)
    print(f"wrote {cb._DATA} ({len(out)} tables, {built} new)")


if __name__ == "__main__":
    main()
