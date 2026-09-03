# The Gridbook trellis rate surface — archived 2026-09-02

**Kill order.** Robert, 2026-09-02: the Gridbook / NVFP4-CB lane is retired
from PrismaQuant and the sanctioned containers are three — `compressed-tensors`
on vanilla vLLM, GGUF, and **Tessera** on Tessera's own vLLM plugin. The
Gridbook lane's producer side went to `archive/gridbook_lane_2026-09-02/`. This
directory is the part that outlived it by a few hours: the **allocator-side
rate surface** that priced trellis rungs against the retired lane's wire.
Not a measurement verdict and not a defect — the wire has no runtime left to
serve it. Tracked as RobTand/prismaquant#118.

**Last commit where these modules were live:** `dfad8d1` (branch
`tessera/decouple-gridbook`). Everything here is that commit's content, moved
with `git mv` so `git log --follow` still reaches it. The archiving commit is
the one on `tessera/decouple-gridbook` citing #118; find it with

    git log --diff-filter=R --oneline -- archive/trellis_wire_2026-09-02

## What the wire was

`gridbook.trellis.wire.v1` (`trellis_formats.TRELLIS_WIRE_SCHEMA`): an
88-byte-header, 16-byte-row-aligned serialization of trellis-coded quantization
over two grids — `TCQ_E2M1` and `TCQ_E4M3` — addressed on a q256 rate axis, so
`TCQ_E2M1_R640` means "2.5 bits per weight in the E2M1 family". 2,546 legal
rung names, each priced in **exact serialized bytes** (`trellis_footprint`),
with a per-tensor rate/distortion hull and a rate surface fitted from measured
anchors (`trellis_allocator`, `trellis_rate_surface`) and offered to the
multi-choice DP through an opt-in seam (`trellis_menu`).

## Why it is retired

Rob's 2026-09-02 lane decision, not a defect: **the byte model is
`gridbook.trellis.wire.v1`, and it is NOT a port of Tessera's.** Tessera's wire
is `prismaquant.tessera.v1` — a different plane set, deliberately not a port
(`prismaquant/tessera_footprint.py` says so in its own header). The two share
the English words "trellis", "rate" and "rung" and nothing else: Tessera
addresses `(base grid, arity)` pairs on a q256-continuous axis, and its
footprint delegates to `tessera.layout.build_planes` rather than to any of the
byte arithmetic here. Nothing of the layout below carries over, which is why
this was archived whole instead of ported.

With the Gridbook lane gone, no sanctioned runtime reads these bytes, so the
surface could only ever have been an allocation-time report — which is exactly
what it already was. It never shipped an artifact: the seam refused when the
flag was set (`trellis_menu.UNWIRED_LINKS`, eight missing links between a
priced menu and a selectable assignment), and the exporter refused any
`TCQ_*_R256` rung that reached it. Both refusals were live and both are gone
with the code; the exporter's generic "no compressed-tensors emit path"
refusal still fails closed on a `TCQ_*` name and still names the format.

## What is here

- `prismaquant/trellis_formats.py` — the wire schema, the two families, the
  2,546 legal rung names, `parse_trellis_format_name`, the layout and schedule
  validators.
- `prismaquant/trellis_footprint.py` — the exact byte model (88-byte header,
  16-byte row alignment, 32-bit block offsets, 4-bit schedule codes, 3-byte
  alphabet directory entries).
- `prismaquant/trellis_allocator.py` — exact-rational RD hulls, λ intervals,
  the profile capability gate, the solver candidate menu.
- `prismaquant/trellis_rate_surface.py` — anchor fitting, log-space
  interpolation, densification, leave-one-anchor-out, allocation regret.
- `prismaquant/trellis_menu.py` — the opt-in production seam and its refusal
  (`PRISMAQUANT_TRELLIS_SURFACE`), the manifest schema, `UNWIRED_LINKS`.
- `prismaquant/serving_profile_specs/trellis_research_sm121.json` — the
  emulation-only profile that existed solely to give `_capability_gate` a
  `target_platform` to compare against. Its live successor is
  `tessera_research_sm121.json`; no assignment migrates between them, because
  the anchors' currency here is a weighted-SSE output-MSE proxy, explicitly not
  the AURA KL-adjoint the production DP ranks in. Archiving it also removes one
  parametrized case from `test_model_profile_conformance.py`
  (`test_every_serving_spec_key_is_parsed[_no_runtime_rationale]`, a skip): that
  documentation-only key existed in no other spec.
- `tests/test_trellis_{allocator,formats,menu,rate_surface}.py` — 84 node IDs
  (73 test functions; the rest are parametrized cases), all
  pinning the retired wire. They are archived alongside the modules rather than
  deleted, so the byte model stays executable if anyone needs to read it.

## What replaced it in the live tree

- **The rate surface machinery**, pointed at Tessera and renamed:
  `tessera_formats.py`, `tessera_footprint.py`, `tessera_allocator.py`,
  `tessera_rate_surface.py`, `tessera_menu.py`.
- **The seam** is gone, not renamed. Tessera's continuous rungs need no flag:
  `allocator_candidates.reduce_continuous_menu` reduces them into every menu
  unconditionally, and it is a no-op on a menu with no Tessera rung in it.
- **`PRISMAQUANT_TRELLIS_SURFACE` still refuses**, from
  `allocator_candidates.refuse_retired_trellis_surface`. Dropping the variable
  would have let a stale driver keep exporting it and get a *different*
  allocation with no diagnostic — a gate that fails open, the failure class of
  prismaquant#120. Pinned by `tests/test_retired_trellis_surface_refusal.py`.

## The durable lesson

A cost model that prices a wire is only as alive as the runtime that reads the
wire. This surface was arithmetically correct, exactly byte-priced, and
thoroughly tested from the day it landed (`2b8d289`, 2026-08-26) — and none of
that mattered once the lane it priced was retired, because a rung the
allocator can price but no runtime can serve is a serving gap being reported,
not a format being offered
(principle 9). The seam was right to refuse, and the honest end state for it
was archival, not a port.
